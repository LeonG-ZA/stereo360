"""Filename tokens that tell a VR player what a file is.

The other half of "how does a player know". GPano says the projection but has
no way to say there are two eyes, so for stereo the **filename** is the signal
— and on a Quest 3 it is sufficient on its own: a stacked frame with no
metadata at all displayed correctly purely because it was called `_360_TB`.

Neither signal is required when the other is present, so this is belt and
braces rather than a decision. Worth having anyway: the tool knows both facts,
and the alternative is expecting everyone to have read a forum thread.

## The conventions

They converge across SKYBOX, DeoVR and HereSphere, are case-insensitive, and
treat any separator (or none) as equivalent:

    projection    360, 180, 360x180, 180x180
    top-bottom    TB, OU, 3DV
    side-by-side  SBS, LR, 3DH
    mono          no stereo token at all

## What gets suggested

`_360_TB` and `_180x180_3dh`, because those are the two that were **measured**
displaying correctly on a Quest 3. Other spellings on the list are equally
documented and were not tested here, and inventing a variant nobody tried is
how you end up debugging a filename.
"""

from __future__ import annotations

import os
import re
from typing import Optional

#: Spellings that mean "two eyes, stacked vertically".
TOP_BOTTOM_TOKENS = ("tb", "ou", "3dv")

#: Spellings that mean "two eyes, side by side".
SIDE_BY_SIDE_TOKENS = ("sbs", "lr", "3dh")

#: What this tool writes, chosen because they were tested rather than merely
#: documented. `{stem}` keeps whatever the user called it.
SUFFIXES = {"360": "_360_TB", "vr180": "_180x180_3dh"}

_STEREO_TOKENS = frozenset(TOP_BOTTOM_TOKENS + SIDE_BY_SIDE_TOKENS)


def _tokens(stem: str) -> list:
    """`garden_360_TB` -> ['garden', '360', 'tb'].

    Split on separators only, never inside a run of characters. Substring
    matching would be a trap: "tb" appears inside "artbook", and a photo
    called `artbook.jpg` is not a stereo pair.

    The cost is that `garden360TB.jpg` -- which the conventions do allow, since
    the separator may be nothing -- is not recognised, so it gets a suggestion
    it does not need. A redundant suggestion is a much cheaper mistake than a
    file that silently keeps a name saying nothing.
    """
    return [t for t in re.split(r"[^0-9A-Za-z]+", stem.lower()) if t]


def describes_stereo(path: str) -> bool:
    """Whether the filename already says how the two eyes are packed.

    Only the *stereo* token counts, not the projection one. GPano carries the
    projection perfectly well, so that is not what the filename is needed for
    -- and looking for `180` would call `IMG_0180.jpg` self-describing, which
    it plainly is not.
    """
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    return bool(_STEREO_TOKENS.intersection(_tokens(stem)))


def suggest(path: str, output_mode: str = "360") -> str:
    """The same path renamed so a player can read it, or unchanged.

    Unchanged when the name already carries a stereo token, so a deliberate
    choice is never argued with.
    """
    suffix = SUFFIXES.get(output_mode)
    if suffix is None or describes_stereo(path):
        return str(path)
    stem, ext = os.path.splitext(str(path))
    return f"{stem}{suffix}{ext}"


def advice(path: str, output_mode: str = "360") -> Optional[str]:
    """A one-line suggestion for this path, or None when it needs none."""
    better = suggest(path, output_mode)
    if better == str(path):
        return None
    return (f"The filename does not say how the eyes are packed. Players that "
            f"read filenames rather than metadata -- SKYBOX, DeoVR, "
            f"HereSphere -- would recognise "
            f"{os.path.basename(better)!r}. The tag already covers players "
            f"that read metadata, so this is belt and braces.")
