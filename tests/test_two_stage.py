"""The factor, the model, and whatever finishes the job.

An upscale was always two steps and only ever looked like one. A 2x model
asked for 4x has always handed the remaining 2x to libplacebo's default
spline36, silently -- so "4x with ArtCNN" meant 2x of ArtCNN and 2x of
spline36, and nothing said so. The interface now asks for the factor first
and shows both stages, which means the arithmetic behind them has to be
right in one place rather than guessed at in two.

The cases are Leon's, written before any of this existed, and they are the
test:

  1. 2K to 8K at 4x with a 2x model   -> an upscaler at 2x
  2. 2K to 8K at 4x with a 4x model   -> nothing left to do
  3. 2K to 8K at 4x with no model     -> an upscaler at 4x
  4. 4K to 8K at 2x with a 4x model   -> a *downscaler* at 0.5x
  5. 4K to 8K at 2x with a 2x model   -> nothing left to do
  6. 4K to 8K at 2x with no model     -> an upscaler at 2x
"""

import pytest

from stereo360 import fsrcnnx, upscalers


def _residual(target, code):
    return upscalers.residual(target, upscalers.get(code) if code else None)


class TestTheSixCases:
    def test_1_a_2x_model_at_4x_leaves_2x(self):
        assert _residual(4.0, "artcnn32") == pytest.approx(2.0)

    def test_2_a_4x_model_at_4x_leaves_nothing(self):
        assert _residual(4.0, "lsdir") == pytest.approx(1.0)

    def test_3_no_model_at_4x_leaves_all_of_it(self):
        assert _residual(4.0, None) == pytest.approx(4.0)

    def test_4_a_4x_model_at_2x_leaves_a_downscale(self):
        """The one case that reverses direction, and so needs a different
        filter set: an area filter averages the samples it discards, where a
        sharpening upscaler run backwards just aliases."""
        assert _residual(2.0, "siax") == pytest.approx(0.5)

    def test_5_a_2x_model_at_2x_leaves_nothing(self):
        assert _residual(2.0, "compactldl") == pytest.approx(1.0)

    def test_6_no_model_at_2x_leaves_all_of_it(self):
        assert _residual(2.0, None) == pytest.approx(2.0)


class TestAnyScale:
    def test_a_model_that_takes_an_output_size_never_leaves_a_remainder(self):
        """NVIDIA VSR is asked for dimensions rather than a factor, so there
        is no mismatch to pay for at any factor it accepts."""
        for target in (1.5, 2.0, 3.0, 4.0):
            assert _residual(target, "nvvsr_ultra") == pytest.approx(1.0)

    def test_and_neither_does_a_resampler(self):
        for target in (2.0, 4.0):
            assert _residual(target, "ewa_lanczos4sharpest") == \
                pytest.approx(1.0)

    def test_a_factor_of_zero_is_refused_rather_than_dividing(self):
        with pytest.raises(ValueError):
            upscalers.residual(0, upscalers.get("artcnn32"))


class TestDirection:
    def test_only_the_area_filter_goes_down(self):
        """The second dropdown shows one list or the other, never both, and
        this is what it filters on."""
        down = [v.code for v in upscalers.in_category(upscalers.RESAMPLER)
                if v.direction == "down"]
        assert down == ["area"]

    def test_every_upward_resampler_names_a_libplacebo_preset(self):
        """`option` is what reaches the filtergraph. Lanczos is the one that
        may leave it empty, because it is libplacebo's own default name."""
        for v in upscalers.in_category(upscalers.RESAMPLER):
            if v.direction != "up":
                continue
            assert v.option or v.code == "lanczos", v.code


class TestTheFilterChain:
    """What the two stages actually become, since the chain is where a
    mistake would be invisible until a render finished."""

    SHADER = "models/ArtCNN_C4F32.glsl"

    def test_a_model_with_a_named_finisher_carries_both(self):
        vf = fsrcnnx.chain(1920, 960, 4.0, shader=self.SHADER,
                           resampler="ewa_lanczos4sharpest")
        assert "w=7680:h=3840" in vf
        assert "upscaler=ewa_lanczos4sharpest" in vf
        assert "custom_shader_path" in vf

    def test_a_model_with_no_finisher_named_carries_only_itself(self):
        """libplacebo's own default then applies, which is what happened
        before any of this was a control."""
        vf = fsrcnnx.chain(3840, 1920, 2.0, shader=self.SHADER)
        assert "upscaler=" not in vf
        assert "custom_shader_path" in vf

    def test_no_model_at_all_is_a_plain_resample(self):
        """Scenarios 3 and 6. NO_SHADER is distinct from None, which still
        means "the default shader" -- conflating them would silently run
        FSRCNNX where someone asked for Lanczos."""
        vf = fsrcnnx.chain(1920, 960, 4.0, shader=fsrcnnx.NO_SHADER,
                           resampler="ewa_lanczos4sharpest")
        assert vf.endswith("w=7680:h=3840:upscaler=ewa_lanczos4sharpest")
        assert "custom_shader_path" not in vf

    def test_none_still_means_the_default_shader(self):
        vf = fsrcnnx.chain(3840, 1920, 2.0, shader=None)
        assert "custom_shader_path" in vf


class TestTheCommandTheUiBuilds:
    def test_the_second_stage_reaches_the_command_line(self):
        from stereo360_ui import options

        argv = options.build_argv(
            {"input": "in.mp4", "output": "out.mp4", "upscale": True,
             "upscaleModel": "artcnn32",
             "upscaleResampler": "ewa_lanczos4sharpest",
             "upscaleScale": 4.0})
        assert "--upscale" in argv and "artcnn32" in argv
        assert "--upscale-resampler" in argv
        assert argv[argv.index("--upscale-resampler") + 1] == \
            "ewa_lanczos4sharpest"

    def test_nothing_is_emitted_when_the_model_covers_the_factor(self):
        """A command that reads as one step where it is one step."""
        from stereo360_ui import options

        argv = options.build_argv(
            {"input": "in.mp4", "output": "out.mp4", "upscale": True,
             "upscaleModel": "compactldl",
             "upscaleResampler": "", "upscaleScale": 2.0})
        assert "--upscale-resampler" not in argv

class TestEveryKindCanActuallyRun:
    """A model the interface offers must have a branch that runs it.

    NVIDIA VSR was selectable for a while and had none: `kind` was "nvvsr",
    no branch matched, and the name fell through to the Topaz lookup, which
    answered "No Topaz upscaling model called 'nvvsr_ultra'" -- true, and
    about the wrong thing entirely. The table grew two new kinds in one
    change and this is what notices a third arriving without a runner.
    """

    def test_every_kind_in_the_table_is_dispatched(self):
        import re

        from stereo360 import upscalers

        source = open("stereo360/cli.py", encoding="utf-8").read()
        dispatched = set(re.findall(r'chosen\.kind == "(\w+)"', source))
        kinds = {v.kind for v in upscalers.VARIANTS}
        missing = kinds - dispatched
        assert not missing, (
            f"{sorted(missing)} appear in the table and nothing in cli.py "
            f"dispatches them -- they would be offered and then refused")

    def test_every_kind_has_a_runner(self):
        """Dispatch alone is not enough; something has to do the work."""
        from stereo360 import upscalers

        runner = {"shader": ("stereo360.fsrcnnx", "run"),
                  "resampler": ("stereo360.fsrcnnx", "run"),
                  "onnx": ("stereo360.esrgan", "run_video"),
                  "nvvsr": ("stereo360.nvvsr", "run")}
        import importlib

        for kind in {v.kind for v in upscalers.VARIANTS}:
            assert kind in runner, f"{kind} has no runner named here"
            mod, fn = runner[kind]
            assert hasattr(importlib.import_module(mod), fn),                 f"{mod}.{fn} is gone, so {kind} cannot run"
