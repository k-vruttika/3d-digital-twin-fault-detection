"""OPC UA server (asyncua): the simulated line as an OPC UA address space.

Address space, in namespace ``namespace_uri``:

    Objects/
      {line_id}/                       (object)
        {machineId}/                   (object, EventNotifier = SubscribeToEvents)
          Speed, Temperature, Vibration, Load, Current, RunTime   (Double; NaN = missing)
          State (String), Seq (Int64), Ts (String, ISO-8601 UTC)
          Start(), Stop(), SetSpeed(value), SetLoad(value), ScheduleMaintenance(), Reset()

Every method takes ``commandId, idempotencyKey, issuedBy`` (String) before its value argument
and returns ``status, reason`` (String) from the simulator's CommandAck.

Each tick also fires one ``ReadingEventType`` event per machine (from the machine node) that
carries all Reading fields. Events are queued per subscriber, so every reading arrives
complete and in order. Variables only expose the latest value, and a client polling them can
miss samples or see a half-updated set. The variables are there for browsing and HMIs.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from asyncua import ua
from asyncua.common.methods import uamethod
from asyncua.common.node import Node
from asyncua.server.server import Server
from asyncua.ua.object_ids import ObjectIds

from twin_common.contracts import Command, CommandAck, ParamValue, Reading
from twin_common.enums import AckStatus, CommandType
from twin_common.logging import get_logger
from twin_common.opcua_names import (
    COMMAND_ARGS,
    COMMAND_METHODS,
    FIELD_BROWSE_NAMES,
    READING_EVENT_TYPE,
)
from twin_sim.publishers.session import SimSession

log = get_logger(__name__)

_VARIANT_TYPES: dict[str, ua.VariantType] = {
    "machine_id": ua.VariantType.String,
    "seq": ua.VariantType.Int64,
    "ts": ua.VariantType.String,
    "state": ua.VariantType.String,
}

# Reading attribute -> (browse name, variant type). Shared by variables and event fields.
FIELDS: tuple[tuple[str, str, ua.VariantType], ...] = tuple(
    (attr, name, _VARIANT_TYPES.get(attr, ua.VariantType.Double))
    for attr, name in FIELD_BROWSE_NAMES.items()
)
VARIABLE_FIELDS = tuple(f for f in FIELDS if f[0] != "machine_id")


def field_value(reading: Reading, attr: str) -> Any:
    value = getattr(reading, attr)
    if attr == "ts":
        return reading.ts.isoformat()
    if attr == "state":
        return reading.state.value
    if value is None:  # missing sensor value
        return math.nan
    return value


def _argument(name: str, vtype: ua.VariantType) -> ua.Argument:
    arg = ua.Argument()
    arg.Name = name
    arg.DataType = ua.NodeId(ua.Int32(vtype.value))
    arg.ValueRank = -1
    return arg


@dataclass
class _MachineNodes:
    obj: Node
    variables: dict[str, Node]
    events: Any  # asyncua EventGenerator


class OpcUaServer:
    def __init__(
        self,
        session: SimSession,
        endpoint: str,
        namespace_uri: str,
        command_timeout_s: float,
    ) -> None:
        self.session = session
        self.endpoint = endpoint
        self.namespace_uri = namespace_uri
        self.command_timeout_s = command_timeout_s
        self.server = Server()
        self._machines: dict[str, _MachineNodes] = {}
        self.ns = 0
        self.events_fired = 0
        self.commands = 0

    async def start(self) -> None:
        server = self.server
        await server.init()
        server.set_endpoint(self.endpoint)
        server.set_server_name("LineTwin simulator")
        server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        self.ns = await server.register_namespace(self.namespace_uri)
        etype = await server.create_custom_event_type(
            self.ns,
            READING_EVENT_TYPE,
            ObjectIds.BaseEventType,
            [(name, vtype) for _, name, vtype in FIELDS],
        )
        line_obj = await server.nodes.objects.add_object(self.ns, self.session.line.line_id)
        for machine_id in self.session.machine_ids:
            obj = await line_obj.add_object(self.ns, machine_id)
            variables: dict[str, Node] = {}
            for attr, name, vtype in VARIABLE_FIELDS:
                initial: Any = (
                    math.nan if vtype is ua.VariantType.Double else (0 if attr == "seq" else "")
                )
                variables[attr] = await obj.add_variable(self.ns, name, ua.Variant(initial, vtype))
            for ctype, (method_name, param) in COMMAND_METHODS.items():
                inputs = [_argument(a, ua.VariantType.String) for a in COMMAND_ARGS]
                if param is not None:
                    inputs.append(_argument("value", ua.VariantType.Double))
                outputs = [
                    _argument("status", ua.VariantType.String),
                    _argument("reason", ua.VariantType.String),
                ]
                await obj.add_method(
                    self.ns,
                    method_name,
                    self._method(machine_id, ctype, param),
                    inputs,
                    outputs,
                )
            await obj.set_event_notifier([ua.EventNotifier.SubscribeToEvents])
            events = await server.get_event_generator(etype, obj.nodeid)
            self._machines[machine_id] = _MachineNodes(obj, variables, events)
        await server.start()
        self.session.add_listener(self._on_reading)
        log.info("opcua.started", endpoint=self.endpoint, machines=self.session.machine_ids)

    async def stop(self) -> None:
        self.session.remove_listener(self._on_reading)
        # asyncua closes the open connections first and the listening socket afterwards. A
        # client that reconnects in between is accepted and never closed, and since Python
        # 3.11.x/3.12 ``Server.wait_closed()`` waits for every connection, so stop() would hang.
        # Stopping the listener first closes that window.
        listener = getattr(getattr(self.server, "bserver", None), "_server", None)
        if listener is not None:
            listener.close()
        await self.server.stop()

    async def _on_reading(self, reading: Reading) -> None:
        nodes = self._machines.get(reading.machine_id)
        if nodes is None:
            return
        for attr, _name, vtype in VARIABLE_FIELDS:
            await nodes.variables[attr].write_value(ua.Variant(field_value(reading, attr), vtype))
        event = nodes.events.event
        for attr, name, _vtype in FIELDS:
            setattr(event, name, field_value(reading, attr))
        event.SourceName = reading.machine_id
        event.Message = ua.LocalizedText(f"{reading.machine_id} seq {reading.seq}")
        await nodes.events.trigger()  # stamps Time = now (UTC), used as sentAt
        self.events_fired += 1

    def _method(
        self, machine_id: str, ctype: CommandType, param: str | None
    ) -> Callable[..., Awaitable[list[str]]]:
        @uamethod  # type: ignore[untyped-decorator,unused-ignore]
        async def handler(
            _parent: Any, command_id: str, key: str, issued_by: str, *value: float
        ) -> list[str]:
            self.commands += 1
            params: dict[str, ParamValue] = {param: float(value[0])} if param and value else {}
            try:
                cmd = Command(
                    command_id=command_id,
                    idempotency_key=key,
                    machine_id=machine_id,
                    type=ctype,
                    params=params,
                    issued_by=issued_by,
                    ts=datetime.now(UTC),
                )
            except ValueError as exc:
                return [AckStatus.REJECTED.value, f"invalid command: {exc}"]
            ack: CommandAck | None = await self.session.submit(cmd, self.command_timeout_s)
            if ack is None:
                return [AckStatus.TIMEOUT.value, "the simulator did not apply the command in time"]
            return [ack.status.value, ack.reason or ""]

        return handler  # type: ignore[no-any-return]
