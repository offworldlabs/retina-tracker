<!--
  Shared security rules for Offworld Labs repos.
  The setup-repo skill copies these into each consuming repo's .claude/rules/ so every
  repo enforces the same baseline. Edit here; the change propagates to every
  repo that copies this file. Replace the TODO placeholders below with real
  imperatives before relying on them.
-->

# Security Rules

- TODO: Never commit secrets, credentials, or API keys — use environment variables or a secrets manager.
- TODO: Validate and sanitise all external input at trust boundaries.
- TODO: Keep dependencies patched and pinned; review new dependencies before adding them.
