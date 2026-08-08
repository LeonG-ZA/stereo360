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
