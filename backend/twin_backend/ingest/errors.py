"""Rejections raised by the normalizer and validator; the pipeline quarantines them."""

from __future__ import annotations

from twin_common.enums import QuarantineReason


class RejectedMessageError(Exception):
    """A message cannot become a canonical Reading."""

    def __init__(
        self,
        reason: QuarantineReason,
        detail: str,
        stage: str,
        machine_id: str | None = None,
    ) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail
        self.stage = stage
        self.machine_id = machine_id
