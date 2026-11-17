"""Channel schema (ADR 0028): the simulator defaults are unchanged and real schemas work."""

from __future__ import annotations

import numpy as np
import pytest

from twin_common.enums import MachineKind
from twin_ml.config import load_dataset_config
from twin_ml.features import inputs
from twin_ml.features.normalize import Normalizer
from twin_ml.features.schema import SIM_SCHEMA, ChannelSchema
from twin_ml.features.summary import summary_features, summary_names
from twin_ml.testing import CONFIG_DIR

# Fingerprints of the Phase 7-9 dataset configs, recorded before the Phase 10 refactor. A change
# would change every dataset key, and with it every stored run's match (ADR 0028).
PINNED = {
    "dataset": "a4a16d97aea89cd9010e8e3976a157a48c423ef7efd7ed2ba75420a55ecffe89",
    "dataset_e4": "b77dac4e0b51b9ffa79b7e3ed939bc52e7baa6cc97b21eb371e27bd71b1f9dc2",
    "dataset_aug": "b12814abe23c1645ad714538623d084fd2d9054c91208ab37a66a858cc3451ab",
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_simulator_dataset_fingerprints_unchanged(name: str) -> None:
    cfg = load_dataset_config(CONFIG_DIR / "models" / f"{name}.yaml")
    assert cfg.fingerprint() == PINNED[name]


def test_sim_schema_equals_the_phase7_constants() -> None:
    assert SIM_SCHEMA.sensors == ("speed", "temp", "vib", "load", "current")
    assert (*SIM_SCHEMA.sensors, "vib_norm") == inputs.VALUE_CHANNELS
    assert tuple(f"{s}_missing" for s in SIM_SCHEMA.sensors) == inputs.MASK_CHANNELS
    assert tuple(f"kind_{k.value}" for k in MachineKind) == inputs.KIND_CHANNELS
    assert len(inputs.SEQUENCE_CHANNELS) == 14
    assert len(summary_names()) == 50
    assert SIM_SCHEMA.step_s == 1.0
    assert ChannelSchema.of_meta({}) == SIM_SCHEMA


def test_schema_validation() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ChannelSchema(sensors=("a", "a"), kinds=("K",))
    with pytest.raises(ValueError, match="vib_norm"):
        ChannelSchema(sensors=("a",), kinds=("K",), vib_norm=True)
    s = ChannelSchema(sensors=("p", "q"), kinds=("K1", "K2"), step_s=60)
    assert s.value_channels == ("p", "q")
    assert s.sequence_channels == ("p", "q", "p_missing", "q_missing", "kind_K1", "kind_K2")
    assert s.steps(3600) == 60
    with pytest.raises(ValueError, match="multiple"):
        s.steps(90)
    assert ChannelSchema.from_dict(s.to_dict()) == s


def test_sim_normalizer_dict_has_no_schema_key() -> None:
    rng = np.random.default_rng(0)
    raw = rng.uniform(1, 10, size=(50, 5)).astype(np.float32)
    norm = Normalizer.fit([(raw, MachineKind.CNC)], 0.1)
    d = norm.to_dict()
    assert "schema" not in d
    assert d["channels"] == list(inputs.VALUE_CHANNELS)
    back = Normalizer.from_dict(d)
    assert back.schema == SIM_SCHEMA
    np.testing.assert_array_equal(
        back.transform(raw, MachineKind.CNC)[0], norm.transform(raw, MachineKind.CNC)[0]
    )


def test_real_schema_normalizer_and_summary() -> None:
    s = ChannelSchema(sensors=("p", "q", "r"), kinds=("A",), step_s=3600)
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(40, 3)).astype(np.float32)
    raw[3, 1] = np.nan
    norm = Normalizer.fit([(raw, "A")], 0.1, schema=s)
    assert norm.speed_ref == {}
    values, mask = norm.transform(raw, "A")
    assert values.shape == (40, 3)
    assert mask[3, 1] == 1
    back = Normalizer.from_dict(norm.to_dict())
    assert back.schema == s
    feats = summary_features(values[None, :10], mask[None, :10], np.ones((1, 1), np.float32), s)
    assert feats.shape == (1, len(summary_names(s))) == (1, 3 * 7 + 3 + 1)
