"""GET /line: the loaded line config (machines, parts, buffers, topology) for the 3D twin."""

from __future__ import annotations

from fastapi import APIRouter

from twin_backend.api.deps import Ctx, Viewer
from twin_backend.api.models import (
    LineBufferModel,
    LineMachineModel,
    LineResponse,
    MachineLayoutModel,
    SensorSpecModel,
    TopologyModel,
)
from twin_backend.engine.topology import Topology
from twin_common.config.line import LineConfig

router = APIRouter(tags=["line"])


def line_response(line: LineConfig) -> LineResponse:
    """The line config as the frontend needs it; topology as the twin engine derives it."""
    topo = Topology.from_line(line)
    return LineResponse(
        line_id=line.line_id,
        sensors={
            name: SensorSpecModel(unit=s.unit, min=s.min, max=s.max)
            for name, s in line.sensors.items()
        },
        machines=[
            LineMachineModel(
                id=m.id,
                kind=m.kind,
                name=m.name,
                sensors=list(m.sensors),
                parts=list(m.parts),
                sensor_parts=dict(m.sensor_parts),
                layout=MachineLayoutModel(
                    position=m.layout.position, rotation_deg=m.layout.rotation_deg
                )
                if m.layout is not None
                else None,
            )
            for m in line.machines
        ],
        buffers=[
            LineBufferModel(id=b.id, from_=b.from_machine, to=b.to_machine, capacity=b.capacity)
            for b in line.buffers
        ],
        topology=TopologyModel(
            order=list(topo.machine_ids),
            upstream=dict(topo.upstream),
            downstream=dict(topo.downstream),
        ),
    )


@router.get(
    "/line",
    response_model=LineResponse,
    summary="Line layout: machines, kinds, parts, buffers and topology from the line config",
)
async def get_line(ctx: Ctx, _: Viewer) -> LineResponse:
    return line_response(ctx.runtime.line)
