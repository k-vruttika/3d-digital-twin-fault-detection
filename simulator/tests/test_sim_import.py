"""Smoke test: package is installed in the workspace and can use twin_common."""

import twin_common
import twin_sim


def test_import() -> None:
    assert twin_sim.__doc__
    assert twin_common.Reading.model_fields
