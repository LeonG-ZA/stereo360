# Depth scoring harness

`score.py` scores a depth map against three artifacts measured on
`indoor.jpg`, each checked against a law the world guarantees rather than
anyone's judgement: a flat floor's inverse depth follows sin(latitude), a flat
wall's follows cos(elevation), and the gaps between chair slats must read as
the wall they show. `damp.py` holds the rejected damping experiments and
`cli_defaults()`, which any script rendering through the library needs.

The findings this produced are in `findings.md`, under "Post-processing cannot
fix the depth model" and "Choosing a depth model". They are there rather than
here because they decided what the tool ships.

Three guards exist because three reviews went out broken and were caught by
eye rather than by score:

- `depth_span`, because the other scores are ratios and could not see the
  stereo collapsing to nothing.
- `cli_defaults()`, because the library's defaults are not the CLI's: a render
  with `gradient_limit` at its library default of 0.0 tears fine structure
  into fragments.
- Validate an edge measurement on the *left* eye first. It is the untouched
  source, so a tracer reporting hundreds of pixels of bend there is measuring
  itself. Two measurements here did exactly that.
