"""One question, one unit: what size do you want out?

The panel used to ask twice. "Amount" was a multiplier on the source and
"Resolution" was a delivered size, side by side, and nothing stopped the two
answers disagreeing -- a 3840 source at 2x delivered at 5760 computed 7680
pixels and threw a quarter of them away before the depth pass ever saw them.
It was already half tied: the amount decided what the resolution list could
contain, but choosing a resolution did not decide the amount.

Now the resolution is the question and the amount follows from it. Which
inverts the list too: upscaling to *less* than the source is a contradiction,
so with a pre-pass on, the sizes below the source are not offered.

Supersample keeps the headroom the tie would otherwise remove. It already
means "render above the delivered size and resize down", and with a pre-pass
in front that is worth having: on the matched SPAN pair, upscaling past the
target and letting an area filter come back down measured 1.5 dB ahead of
upscaling straight to it.
"""

import pytest

from stereo360_ui import options


class TestTheListInverts:
    def test_without_a_prepass_it_runs_down_from_the_source(self):
        got = options.resolution_choices(3840, 1920, "360")
        widths = [r["width"] for r in got]
        assert widths[0] == 3840, "the source is the master and comes first"
        assert all(w <= 3840 for w in widths)

    def test_with_one_it_runs_up_from_the_source(self):
        got = options.resolution_choices(3840, 1920, "360", upscaling=True)
        widths = [r["width"] for r in got]
        assert widths == [7680, 5760, 4096]
        assert all(w > 3840 for w in widths), (
            "upscaling to less than the source is a contradiction and must "
            "not be offerable")

    def test_it_stops_at_what_an_upscaler_will_do(self):
        """`--upscale-scale` takes up to 4x, so nothing beyond that can be
        delivered however standard the width looks."""
        got = options.resolution_choices(1920, 960, "360", upscaling=True)
        widths = [r["width"] for r in got]
        assert widths, "a 1080p source has plenty of room above it"
        assert max(widths) <= 1920 * options.MAX_UPSCALE

    def test_a_source_at_the_ceiling_has_nowhere_to_go(self):
        """An 8K source cannot be upscaled to a standard width, and the
        interface should offer nothing rather than something impossible."""
        assert options.resolution_choices(7680, 3840, "360",
                                          upscaling=True) == []

    def test_every_entry_carries_the_factor_it_implies(self):
        """So the panel can say "1.50x" without the reader multiplying."""
        for r in options.resolution_choices(3840, 1920, "360", upscaling=True):
            assert r["scale"] == pytest.approx(r["width"] / 3840, abs=1e-4), (
                "stored to four places, so compared to four places")


class TestLeonsCase:
    """A 3840 source delivered at 5760, which is what started this."""

    SOURCE = 3840

    def test_the_matched_factor_is_one_and_a_half(self):
        got = {r["width"]: r["scale"]
               for r in options.resolution_choices(self.SOURCE, 1920, "360",
                                                   upscaling=True)}
        assert got[5760] == pytest.approx(1.5)

    def test_5760_is_unreachable_without_a_prepass(self):
        """Which is why the upscale switch is what makes the target legal at
        all -- the pipeline refuses to deliver wider than its source."""
        from stereo360 import pipeline

        with pytest.raises(ValueError):
            pipeline.scaled_eye_size(self.SOURCE, 1920, 5760)

    def test_and_reachable_with_one(self):
        from stereo360 import pipeline

        up_w, up_h = int(self.SOURCE * 1.5), int(1920 * 1.5)
        assert pipeline.output_geometry(up_w, up_h, "360", 5760) == (5760, 5760)


class TestTheCommandItBuilds:
    def test_the_matched_factor_reaches_the_command_line(self):
        argv = options.build_argv(
            {"input": "in.mp4", "output": "out.mp4", "upscale": True,
             "upscaleModel": "artcnn32", "upscaleScale": 1.5,
             "outputWidth": 5760})
        assert "--upscale-scale" in argv
        assert argv[argv.index("--upscale-scale") + 1] == "1.5"
        assert "--output-width" in argv
        assert argv[argv.index("--output-width") + 1] == "5760"

    def test_a_factor_of_two_is_the_default_and_says_nothing(self):
        """Left off the command line when it matches the default, the same
        rule every other option follows."""
        argv = options.build_argv(
            {"input": "in.mp4", "output": "out.mp4", "upscale": True,
             "upscaleModel": "artcnn32", "upscaleScale": 2.0})
        assert "--upscale-scale" not in argv
