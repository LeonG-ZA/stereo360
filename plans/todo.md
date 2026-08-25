# TODO

Work that is understood and wanted but not yet built. Each entry says what it
is, why it earns its place, and what the evidence for it was — so it can be
picked up cold.

## Done: choose which eye is the source, and how far off it the other sits

Built as one number, `left_share`, rather than the two controls this entry
originally described. The share and the side are not independent -- "which
eye is the source" is just which end of the range you are at -- so one
parameter covers both and there is a single code path instead of two.

* `stereo360/pipeline.py` -- `stereo_pair(frame, disp, strength,
  left_share=0.0)`. 0 leaves the left eye untouched, 0.5 splits evenly, 1
  leaves the right eye untouched. The video path builds one stream at the
  extremes and two in between, and `_eyes_warped` tells the chunk sizer
  which, since that is what chunk memory scales with.
* `stereo360/cli.py` -- `--left-share F` and `--source-eye {left,right}`,
  resolved by `_left_share`. `--split-baseline` still works and means 0.5.
* `stereo360_ui/` -- a "Sharp eye" combo and a "Baseline shared" slider,
  which write one derived `leftShare`. Presets saved with the old boolean
  still open and still mean an even split.

Verified: total disparity is conserved across every share -- measured -9.00
to -9.50 px against a geometric ideal of -9.29, within 3.1% -- the eyes move
in proportion to their share, and the source eye is bit-identical to the
input at both extremes.

**Still to decide: the default.** It is 0.0, which is what every earlier
version did, and the evidence for moving it is mixed. On three small
features an even split scored 0.09-1.27 for eye-to-eye disagreement against
8.8-12.3 for the whole baseline in one eye, with a knee suggesting 0.15
recovers most of it. But the floor under the indoor table was still badly
asymmetric at 0.15 (+0.93 / -5.27 px) and only symmetric at 0.5 (+/-3.10),
so the knee may hold for small isolated features and not for extended
surfaces. That wants headset time across a few scenes, not another metric.

## Withdrawn: "the mesh renderer leaves cuts unfilled"

There was an entry here claiming the mesh path fails to fill some of what it
cuts. It was wrong and is kept only so the claim is not repeated.

`fill_holes` fills the whole mask: handed 0.1378% of an indoor frame in blobs
up to 11800 px, it leaves nothing, in both its directional and Telea modes.
The pixels that looked unfilled sit on no cut at all -- they are near-black
scene content, value 7 where the source is 12 and its surroundings 14, which
crossed an arbitrary "darker than 8" line after resampling. At a threshold
that catches genuine holes (value 0) an indoor mesh render measures 0.0000%.

The black that really was there came from the measurement scripts, which skip
filling on purpose so it cannot contaminate a shape residual. Assembling their
output into viewable images was the mistake, and it was in the assembly, not
the renderer.

The mesh renderer's real open defects are the ones in its own docstring: a
noise floor the splat does not have, and four hairlines at longitudes +/-45
and +/-135 degrees. Both come from scattering rounded samples instead of
computing coverage per output pixel, and both would go with a proper
rasteriser -- which is also the fast one, since profiling puts the actual
geometry at 1.6% of the cost.
