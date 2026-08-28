"""The warp says where it runs.

The depth banner covers half a frame. The warp is most of the other half and
made its own device decision in silence, so a GPU depth runtime beside a
CPU-only torch printed "GPU accelerated" and was believed while the majority
of every frame ran on the processor. That shipped, on an RTX 5070 Ti.
"""

import pytest

from stereo360 import cli


class _Rec:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, message, **fields):
        self.infos.append((message, fields))

    def warning(self, message, **fields):
        self.warnings.append((message, fields))


def test_a_gpu_warp_is_announced(monkeypatch):
    from stereo360 import warp

    monkeypatch.setattr(warp, "gpu_device", lambda: "cuda")
    rec = _Rec()
    cli._report_warp_device(rec)

    assert not rec.warnings
    assert len(rec.infos) == 1
    message, fields = rec.infos[0]
    assert "warp" in message and "cuda" in message
    assert fields["device"] == "cuda"


def test_a_cpu_warp_is_a_warning_not_an_info(monkeypatch):
    """It has to be loud. The whole failure was that it was silent while a
    correct, reassuring GPU line sat directly above it."""
    from stereo360 import warp

    monkeypatch.setattr(warp, "gpu_device", lambda: None)
    rec = _Rec()
    cli._report_warp_device(rec)

    assert not rec.infos
    assert len(rec.warnings) == 1
    message, fields = rec.warnings[0]
    assert "WARNING" in message
    assert "warp" in message.lower()
    assert fields["device"] == "cpu"
    # says the depth line above does not cover this, which is the confusion
    # that made the original failure invisible
    assert "depth" in message.lower()
    assert "%%" not in message, "literal %% reaches the user"


def test_an_impossible_gpu_request_fails_here_rather_than_mid_render(monkeypatch):
    """STEREO360_GPU_WARP=1 with no device raises. Deliberately not caught:
    failing before frame one beats failing hours in."""
    from stereo360 import warp

    def boom():
        raise RuntimeError("STEREO360_GPU_WARP=1 but torch reports no device")

    monkeypatch.setattr(warp, "gpu_device", boom)
    with pytest.raises(RuntimeError):
        cli._report_warp_device(_Rec())
