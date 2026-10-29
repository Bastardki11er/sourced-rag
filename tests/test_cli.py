import pytest

from sourced.cli import main


def test_cli_ingest_reports_invalid_path_and_closes_index(make_pipeline, tmp_path, monkeypatch, capsys):
    pipeline = make_pipeline()
    monkeypatch.setattr("sourced.pipeline.build_pipeline", lambda *args, **kwargs: pipeline)
    closed = []
    close = pipeline.index.close

    def record_close():
        closed.append(True)
        close()

    monkeypatch.setattr(pipeline.index, "close", record_close)
    with pytest.raises(SystemExit) as exc:
        main(["ingest", str(tmp_path / "missing")])
    assert exc.value.code == 2
    assert "missing" in capsys.readouterr().err
    assert closed == [True]


def test_cli_invalid_settings_have_readable_error(monkeypatch, capsys):
    monkeypatch.setenv("SOURCED_TOP_K", "0")
    with pytest.raises(SystemExit) as exc:
        main(["search", "policy"])
    assert exc.value.code == 2
    assert "SOURCED_TOP_K" in capsys.readouterr().err
