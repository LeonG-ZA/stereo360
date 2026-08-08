# Confidence-weighted disparity damping

Not part of the pipeline. This is the working evidence for whether the idea
is worth building, kept so the numbers can be re-derived rather than taken on
trust.

## The problem

Five artifacts reported against `indoor.jpg`, all one fault: **the depth map
is smooth where the world is sharp, and lumpy where the world is flat.**

| observed | what the depth map does |
|---|---|
| kitchen wall trim vanishes in the right eye | thin feature overtaken by the near surface sweeping across it |
| pillar trim far too wide | the same feature stretched instead, opposite ramp direction |
| wall by the entrance door is not straight | disparity wanders 62% peak to peak down a flat wall |
| chair is uncomfortable to look at | see-through gaps between the slats filled in at 2.5x too near |
| mop on the vacuum reads wrong | pad and shell merged into one smooth dome |

## What does not work

Measured against the chair-gap ratio, all giving the same 1.8x:

- more tiles (`--depth-tiles`)
- model input resolution at 518, 1036, 1456 and 1820
- guided filtering (`--smooth` 8 and 24)
- the Large depth model

It is not a resolution problem. The model does not fail to *see* the gaps; it
decides the chair back is solid. That is a learned prior, and no amount of
pixels changes a semantic judgement.

`--strength` cannot help either, and this is provable rather than measured:
all three scores are ratios, so they are invariant under any uniform scaling
of disparity. Turning the stereo down makes the errors less visible without
making them smaller.

## What does

Two depth estimates disagree where the model is guessing. Damping disparity
*toward far* where they disagree -- one-sided, because these errors are
one-sided: the gaps, the floor ridge and the wall's lower half all read too
near.

| | chair gap (-> 1.0) | wall wobble (-> 0%) | floor rms (-> 0%) |
|---|---|---|---|
| baseline | 1.76 | 103.2 | 40.7 |
| any `--strength` | 1.76 | 103.2 | 40.7 |
| damped, both sources, 0.6 | **1.29** | **49.6** | **35.5** |

Two sources of disagreement, and they catch different things: a second pass
with different tiling is better for the wall and floor, a second pass on the
mirrored image is better for the chair gaps. Mirroring is the cleaner signal
-- same settings, so disagreement is the model being unsure rather than a
change of configuration.

Damping past about 0.8 makes the wall worse again, and at 1.0 the chair gaps
overshoot to 0.94: too far rather than too near.

## Open

- Not yet judged in a headset. The scores are all scale-invariant ratios with
  a physical law behind them, which is more than could be said for several
  measurements in this investigation -- but the headset decides.
- Costs a second depth pass.
- One scene. Needs the tram footage and an outdoor still before it means
  anything general.

## Result: the damping idea does not work

Two formulations, both rejected.

**Scale depth by confidence.** Improves the three ratios but takes 38% of the
depth range with it, and the loss lands in the near field -- confidence is
anti-correlated with depth at -0.59, so near surfaces are damped hardest,
which is the opposite of where anyone wants to lose separation. It also turns
every straight line to sawtooth in the right eye, and there is a reason it
must: d(d*f) = f.dd + d.df, so multiplying by a spatially varying factor
*injects* gradient wherever confidence varies. Confidence varies at object
boundaries. It manufactures depth discontinuities at exactly the edges it is
supposed to be protecting.

**Limit the depth gradient where confidence is low.** No effect at all, at
any floor or iteration count. The reason is worth keeping: these artifacts
are not high-frequency depth noise. The chair's gaps are wrong across their
whole extent and the wall is wrong over hundreds of pixels -- regional depth
that is confidently incorrect. No local operator reaches that, which is also
why guided filtering and smoothing did nothing.

## What did help, slightly

Not damping at all, just using a better depth map. A pass on the mirrored
image scores better than the normal pass on every artifact while keeping the
depth range:

| | chair gap | wall wobble | floor rms | depth span |
|---|---|---|---|---|
| normal pass | 1.76 | 103.2% | 40.7% | 9.26 |
| mirrored pass | 1.71 | 74.0% | 39.7% | 9.13 |
| per-pixel min of three passes | 1.71 | 58.7% | 43.4% | 8.40 |

Ordinary test-time augmentation, in other words. It costs a second depth pass
and gives maybe a quarter off the wall wobble for nothing. One image, so it
could be luck; it would need the tram and an outdoor still before it meant
anything.

## For anyone picking this up

Three guards exist here because three reviews went out broken and were caught
by eye rather than by score:

- `depth_span`, because the other scores are ratios and could not see the
  stereo collapsing to nothing.
- `cli_defaults()`, because the library's defaults are not the CLI's and a
  render with `gradient_limit` at its library default of 0.0 tears fine
  structure into fragments.
- Validate an edge measurement on the *left* eye first. It is the untouched
  source, so a tracer reporting hundreds of pixels of bend there is measuring
  itself. Two of the measurements in this investigation did exactly that.
