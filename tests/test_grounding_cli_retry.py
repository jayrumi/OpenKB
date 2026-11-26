"""A semantic publication refusal is not permission to rerun compilation."""

import pytest


@pytest.mark.parametrize("wrapped", [False, True])
def test_cli_never_replays_a_grounding_refusal(monkeypatch, wrapped):
    import openkb.cli as cli
    from openkb.grounding import GroundingError

    calls = []
    sleeps = []

    async def compile_once():
        calls.append("compile")
        try:
            raise GroundingError("manual source review required")
        except GroundingError as error:
            if wrapped:
                raise RuntimeError("compilation refused") from error
            raise

    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError):
        cli._run_compile_with_retry(compile_once, "test")
    assert calls == ["compile"]
    assert sleeps == []
