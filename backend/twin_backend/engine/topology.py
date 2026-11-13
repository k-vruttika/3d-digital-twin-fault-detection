"""Line topology derived from the line config: who feeds whom.

Buffers define the material-flow links (``from -> to``). A line without buffers is taken in
the machine order of the config (material-flow order, see ``LineConfig.machines``), which is
also how the simulator feeds a single station. The start of the line is fed by ``SOURCE`` and
the end delivers to ``SINK``, as in the simulator (``twin_sim.line``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from twin_common.config.line import LineConfig

SOURCE_ID = "SOURCE"
SINK_ID = "SINK"


@dataclass(frozen=True, slots=True)
class Topology:
    machine_ids: tuple[str, ...]
    upstream: dict[str, str | None]  # None = fed by SOURCE
    downstream: dict[str, str | None]  # None = delivers to SINK

    @classmethod
    def from_line(cls, line: LineConfig) -> Topology:
        ids = tuple(m.id for m in line.machines)
        up: dict[str, str | None] = dict.fromkeys(ids)
        down: dict[str, str | None] = dict.fromkeys(ids)
        if line.buffers:
            for buf in line.buffers:
                down[buf.from_machine] = buf.to_machine
                up[buf.to_machine] = buf.from_machine
        else:
            for a, b in itertools.pairwise(ids):
                down[a] = b
                up[b] = a
        return cls(ids, up, down)

    def neighbour_for(self, machine_id: str, starved: bool) -> str | None:
        """The machine a STARVED (upstream) or BLOCKED (downstream) machine waits for."""
        return self.upstream[machine_id] if starved else self.downstream[machine_id]
