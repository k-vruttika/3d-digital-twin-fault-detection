from __future__ import annotations

from pathlib import Path

import pytest

from twin_common.config.settings import TwinSettings

PG_VARS = (
    "LINETWIN_PG_HOST",
    "LINETWIN_PG_PORT",
    "LINETWIN_PG_DB",
    "LINETWIN_PG_USER",
    "LINETWIN_PG_PASSWORD",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PG_VARS:
        monkeypatch.delenv(var, raising=False)


def _settings(env_file: Path | None) -> TwinSettings:
    return TwinSettings(_env_file=env_file)


def test_defaults_match_compose(tmp_path: Path) -> None:
    s = _settings(tmp_path / "missing.env")
    assert (s.pg_host, s.pg_port, s.pg_db, s.pg_user) == ("127.0.0.1", 5432, "linetwin", "linetwin")
    assert s.pg_password.get_secret_value() == "linetwin-dev"
    assert s.pg_dsn == "postgresql://linetwin:linetwin-dev@127.0.0.1:5432/linetwin"
    assert s.database_url.startswith("postgresql+asyncpg://linetwin:")
    assert "linetwin-dev" not in repr(s)  # SecretStr hides the password


def test_env_file_and_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "LINETWIN_PG_PORT=5433\nLINETWIN_PG_PASSWORD=p@ss/word\nPOSTGRES_DB=ignored\n",
        encoding="utf-8",
    )
    s = _settings(env)
    assert s.pg_port == 5433
    assert s.pg_dsn == "postgresql://linetwin:p%40ss%2Fword@127.0.0.1:5433/linetwin"
    monkeypatch.setenv("LINETWIN_PG_PORT", "6000")  # real environment wins over .env
    assert _settings(env).pg_port == 6000


def test_env_example_matches_defaults() -> None:
    example = Path(__file__).resolve().parents[2] / ".env.example"
    s = _settings(example)
    assert (s.pg_port, s.pg_db, s.pg_user) == (5432, "linetwin", "linetwin")
    assert s.pg_password.get_secret_value() == "linetwin-dev"
