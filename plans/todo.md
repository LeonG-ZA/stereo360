# TODO

Work that is understood and wanted but not yet built. Each entry says what it
is, why it earns its place, and what the evidence for it was — so it can be
picked up cold.

## Choose which eye is the source, and how far off it the other sits

**What.** Two related controls, one number each:

* which eye keeps the source — the left, as now, or the right;
* what share of the separation that eye takes, from 0 (untouched source) to
  0.5 (an even split).

Both are already possible in the warp: `right_eye_from_disparity` multiplies
its `strength` by `_BASELINE_SCALE`, so passing `-f * total` and
`(1 - f) * total` puts the eyes at those shares, and swapping the signs swaps
which eye is the source. Only `stereo_pair` needs widening — it currently
offers whole-baseline or an even split and nothing between.

**Why it is scene-dependent, and therefore a control rather than a default.**
Which eye is the source decides, for every occluding edge in the scene,
whether the warped eye *hides* what is behind it or *reveals* it. Hiding costs
nothing: the geometry simply covers something up. Revealing has to be invented.
And which way round that falls depends on where each occluder sits relative to
the camera, so no single choice is right for every frame.

The indoor scene shows both halves of it. With the source as the left eye, the
generated right eye slides the stone pillar *over* the kitchen tap, so the tap
never has to be invented. With the source as the right eye, the generated left
eye has to reconstruct what lies behind that wall, which it cannot know. On the
same frame the near chair's top rail goes the other way: the warped eye chews
its lower edge ragged, so whichever eye is warped is the one that damages the
rail. A wall at a different angle in another scene reverses both.

**Evidence.** Measured on the near chair's top rail, indoors at 40 mm and a
15/85 share:

| | left eye | right eye |
|---|---|---|
| source near the left | intact (~50 px) | ragged |
| source near the right | ragged | intact (~50 px) |

And on the share itself, measured as how far the two eyes' distortion differs
on three small features, the whole separation in one eye scores 8.8 to 12.3
while an even split scores 0.09 to 1.27. The curve between them has a knee
rather than a slope: a 15% share already recovers most of the agreement. So
the useful range is roughly 0.15 to 0.5, and 0 is only for someone who
deliberately wants one eye untouched.

**Where it goes.**

* `stereo360/pipeline.py` — `stereo_pair` takes a share and a side instead of
  a boolean `split`.
* `stereo360/cli.py` — a flag pair, e.g. `--source-eye {left,right}` and
  `--left-share F`, with the current behaviour as the default so nothing
  existing changes.
* `stereo360_ui/` — a combo box for the side and a slider for the share, in
  `qml/Main.qml`, with defaults in `options.py` (`_DEFAULTS`) and its argv
  emission, plus the dump list in `app.py`. The angular-correction slider added
  earlier is the pattern to copy.

**Caveat worth stating in the UI text.** This chooses *where* the error lands,
not whether there is one. At any share below 0.5 one eye is more correct than
the other, which is the mismatch that tires a viewer; at 0.5 both eyes carry
it equally instead. It is a trade between a sharper eye and a better-agreeing
pair.

## Fill the mesh renderer's cuts properly

The mesh path cuts geometry at silhouettes and then has to fill what it
removed, and the fill does not reach everything: 0.012% to 0.025% of an indoor
frame is left unfilled, with blobs up to about 2000 px. The splat path has no
such problem because `gradient_limit` keeps the warp injective so holes never
form — it pays for that by flattening small structures, which is the whole
reason the mesh exists.

`fill_holes` was built for splat holes, which are thin ribbons along a depth
edge. Mesh cuts are wider and shaped differently. Whether that is the reason
has not been established.

Until it is fixed the mesh path cannot be judged fairly on anything else,
because the residual black dominates the impression.
