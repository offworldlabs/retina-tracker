"""Vulture dead-code whitelist.

Read by tools/check-dead-code.sh. Every name here is one vulture reports as
dead but which must not be deleted — or which nobody has decided about yet.
The distinction matters, so the two live in separate sections.

Add to CONTRACTS only when the name is genuinely referenced by something
vulture cannot see: a framework calling in, a wire format, a config key. Real
dead code should be deleted, not whitelisted.

The UNREVIEWED section is a backlog, not an exemption. Each entry is code that
appears genuinely unreachable and needs a decision — delete it, or wire up
whatever was left unfinished. The gate is green with these listed so that it
starts catching NEW dead code immediately; working through them is separate.
"""
# ruff: noqa: B018, F821
# B018 — bare-name expressions are how vulture whitelists work.
# F821 — these names are defined in other modules; only vulture reads this file.

_ = type("_", (), {})()

# ── Contracts: referenced by something vulture cannot see ─────────────────────
# (none)

# ── UNREVIEWED: appears dead, needs a decision (delete, or finish wiring) ──────
# TODO: no reference found anywhere in the estate
#   retina_tracker/track.py:36  (unused variable)
ASSOCIATED
# TODO: no reference found anywhere in the estate
#   retina_tracker/config.py:179  (unused variable)
SPOOF_POSITION_EPSILON_DEG
# TODO: no reference found anywhere in the estate
#   retina_tracker/kalman.py:24  (unused attribute)
_.dim_meas
# TODO: no reference found anywhere in the estate
#   retina_tracker/tracker.py:380  (unused method)
_.get_all_tracks
# TODO: no reference found anywhere in the estate
#   retina_tracker/tracker.py:374  (unused method)
_.get_tracks
# TODO: no reference found anywhere in the estate
#   retina_tracker/track.py:760  (unused method)
_.is_high_quality
