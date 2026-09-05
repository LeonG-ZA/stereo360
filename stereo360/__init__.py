"""stereo360 — convert monoscopic 360 video to stereoscopic top-bottom 360 video."""

#: Kept in step with the git tag from v1.0.1 onward, and the reason that has a
#: starting point is that it was not true before it.
#:
#: v1.0.0 shipped with this constant still reading 0.1.0. Nothing consulted it,
#: so nothing broke and nobody noticed -- there was no `--version`, the
#: interface never showed it, and the installer recorded the release tag from
#: GitHub instead. It only became a problem once something needed to ask "what
#: is installed here, and is it current?".
#:
#: So: 0.1.0 *is* v1.0.0, and `released_as` says so rather than leaving every
#: caller to know it. Nothing else needs mapping, and nothing else should be
#: added to that table -- the point of matching the tag is that this is the
#: last entry it ever needs.
#:
#: The rule then lapsed twice. v1.0.2 and v1.0.3 were both tagged while this
#: still read 1.0.1, so both releases ship a build that calls itself 1.0.1 --
#: found when a fresh install of v1.0.3 answered `--version` with 1.0.1. That
#: is not repairable from here and deliberately is not attempted:
#: `_RELEASED_AS` is keyed by package version, and all three releases share
#: the same one, so no entry could tell them apart. An install reporting
#: 1.0.1 is v1.0.1, v1.0.2 or v1.0.3, and only the manifest's release tag
#: distinguishes them.
#:
#: `test_no_tag_already_claims_this_version` caught this the moment v1.0.2 was
#: cut and kept saying so through v1.0.3. The guard worked; it was the acting
#: on it that did not. So this holds the *next* release, never the last one.
__version__ = "1.0.5"

#: Package versions from before the two were kept in step, and the release each
#: one actually went out as.
_RELEASED_AS = {"0.1.0": "1.0.0"}


def released_as(version: str = __version__) -> str:
    """The release name for a package version.

    Takes an argument because the interesting case is not this build: it is a
    version read off an install that is already on disk, which may predate the
    rule.
    """
    return _RELEASED_AS.get(version, version)
