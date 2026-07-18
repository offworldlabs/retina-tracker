<!--
  Shared security rules for Offworld Labs repos, adapted for retina-tracker.
-->

# Security Rules

- Never commit secrets, credentials, or radar/node URLs — use environment variables or `.env` (git-ignored).
- Validate and sanitise external detection and ADS-B input at node/trust boundaries before it reaches the tracker.
- Keep dependencies patched and pinned; review new dependencies before adding them.
