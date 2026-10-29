"""Settings read from the environment: a bad value stops the command before any work."""

import pytest

from conftest import index_state
from sourced import cli, pipeline
from sourced.config import Settings


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "1.5", ""])
def test_a_bad_chunk_size_in_the_environment_is_refused(monkeypatch, raw):
    monkeypatch.setenv("SOURCED_MAX_CHUNK_CHARS", raw)

    with pytest.raises(ValueError, match="SOURCED_MAX_CHUNK_CHARS"):
        Settings.from_env()


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_settings_refuse_a_bad_chunk_size(bad):
    with pytest.raises(ValueError, match="max_chunk_chars"):
        Settings(max_chunk_chars=bad)


def test_ingest_with_a_bad_chunk_size_exits_before_loading_anything(handbook_pipeline, monkeypatch, capsys):
    path = handbook_pipeline.settings.index_path
    before = index_state(path)

    def no_pipeline(*args, **kwargs):
        raise AssertionError("the pipeline was built")

    monkeypatch.setattr(pipeline, "build_pipeline", no_pipeline)
    monkeypatch.setenv("SOURCED_INDEX", str(path))
    monkeypatch.setenv("SOURCED_MAX_CHUNK_CHARS", "0")

    assert cli.main(["ingest", "data/handbook"]) == 2
    assert "SOURCED_MAX_CHUNK_CHARS" in capsys.readouterr().err
    assert index_state(path) == before
