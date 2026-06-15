"""Offline player tracking via tracklet collection + post-hoc clustering.

This is the committed re-ID approach (see CLAUDE.md "Direction"): rather than
assigning IDs greedily per frame, we run a tracker over the whole video to get
short tracklets, embed each tracklet's appearance, then cluster tracklets into a
small set of stable identities. Identity is therefore resolved globally with the
whole video in hand — never propagated across a contaminating frame-by-frame chain.

Public entry point: `offline_track.pipeline.run`.
"""

from .pipeline import run  # noqa: F401
