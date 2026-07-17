# Design: Pilot `core:setup-repo` in retina-tracker (issue #5)

Date: 2026-07-17
Tracking: offworldlabs/claude-shared#5

## Objective

Land the Offworld Labs standard repo setup on `retina-tracker` by running the
`core:setup-repo` skill and reconciling everything the skill leaves untouched.
Prove the setup end-to-end by getting a real `claude[bot]` review comment on a
PR, and file any friction discovered back to `claude-shared`.

Hard constraint: **no pre-existing file is clobbered.** The scaffolding engine
(`scaffold-repo.sh`) already guarantees this — it skips any path that exists — so
the substantive work of this pilot is reconciling the *skipped* files, not the
copy itself.

## Context: why this repo is an unusual pilot

`retina-tracker` is *partially* standardized already, so the skill's engine will
write only the missing files and skip six that exist. The skipped set is exactly
where the real work lives:

- The two Claude workflows already exist but ship `pull-requests: read` /
  `issues: read` — the "green review, no comment" bug from the runbook
  (`docs/runbooks/github-actions-claude-review.md`).
- `.gitignore` ignores the entire `.claude/` directory, which would silently
  prevent the scaffolded `.claude/settings.json` and `.claude/rules/*` from ever
  being committed.
- An open PR (#12) already fixes the review workflow — and does so more
  completely than the `core` template.

## File plan

### Scaffold writes (6 missing files — engine handles, no conflict)

| File | Source |
| --- | --- |
| `.claude/settings.json` | marketplace `offworld` + `core@offworld` enabled |
| `.claude/rules/security.md` | shared rules template |
| `.claude/rules/code-style.md` | shared rules template |
| `.editorconfig` | shared editorconfig |
| `.github/workflows/ci.yml` | uv-based ruff lint + format-check + pytest |
| `tests/.gitkeep` | harmless; `tests/` is already populated |

### Reconciliations (engine skips these; done by hand)

| File | State | Action |
| --- | --- | --- |
| `.gitignore` | ignores all of `.claude/` | Un-ignore `.claude/settings.json` and `.claude/rules/`; keep `.claude/settings.local.json` ignored (personal) |
| `.claude/rules/security.md`, `code-style.md` | written as TODO placeholders | Replace TODOs with this repo's real conventions, sourced from existing `CLAUDE.md` "Code Style Guidelines" |
| `.github/workflows/claude-code-review.yml` | `read` perms | `read`→`write` **plus** graft PR #12's `track_progress: true` and `--allowedTools` (inline-comment MCP + `gh` commands) |
| `.github/workflows/claude.yml` | `read` perms | `read`→`write` (PR #12 did not touch this file) |
| `CLAUDE.md` | 161 lines, good | Keep content; append the template's "Org-Wide Context" pointer to shared docs |
| `pyproject.toml` | ruff config matches; `target-version = "py310"` | Set ruff `target-version = "py312"` to match CI Python; **keep `requires-python >= "3.10"`** (tracker runs on ARM/edge where 3.10 matters) |
| `requirements.txt` | real deps (numpy/scipy/pyyaml) | Keep as-is |
| `requirements-dev.txt` | `ruff`, `pytest` | Keep as-is (already matches template) |

## Workflow reconciliation — the crux

The bare `core` asset `claude-code-review.yml` only grants write perms. PR #12's
history proves that on this plugin stack, write perms **alone** do not post a
comment: the `code-review` plugin buffers via
`mcp__github_inline_comment__create_inline_comment`, which must be allow-listed.
So the correct target workflow is:

- `core` structure and comments, plus
- `pull-requests: write` / `issues: write`, plus
- `track_progress: true`, plus
- `--allowedTools "mcp__github_inline_comment__create_inline_comment,Bash(gh pr comment:*),Bash(gh pr diff:*),Bash(gh pr view:*)"`

`claude.yml` gets the perms flip only (its `@claude` responses use the same
write grant; it needs no `allowedTools` because it runs the comment's own
instructions).

**PR #12 is the vehicle.** Rather than open a competing PR, extend PR #12's
branch (`fix/claude-review-comments`) with the remaining full-alignment changes
(the `claude.yml` perms flip, the scaffolded files, `.gitignore`, rules, CLAUDE.md
pointer, pyproject) so the whole standardization lands as one reviewed change.

### Trigger / security decision (public repo)

`retina-tracker` is public. `pull-requests: write` on a `pull_request` trigger is
fork-exploitable in principle, but fork PR workflow runs require maintainer
approval, so the standard `pull_request` trigger is kept for the pilot. Hardening
(`pull_request_target` + author allow-list) is filed as a follow-up rather than
done here.

## Verification (respecting the runbook gotcha)

Prerequisites are already satisfied: `CLAUDE_CODE_OAUTH_TOKEN` secret exists
(2026-01-21), the Claude GitHub App is installed, and workflow runs are green.

`claude-code-action` refuses to run when a PR branch's workflow differs from the
copy on the default branch (a secret-exfiltration guard). Therefore verification
**cannot** happen on the workflow-fix PR itself. Sequence:

1. Merge the standardization PR (extended #12) to `main`.
2. Open a **separate** verification PR that touches **no** `.github/workflows/`
   file (a trivial code or docs change).
3. Confirm a `claude[bot]` review comment posts.
4. Read the run log and confirm `permission_denials_count: 0`.

## Skill-friction follow-ups (file on `claude-shared`)

1. `core` asset `claude-code-review.yml` lacks `track_progress` / `allowedTools`,
   so a repo that adopts it verbatim can still post no comment. Propose
   upstreaming PR #12's additions into the template.
2. `setup-repo` does not detect a `.gitignore` that ignores `.claude/`, so the
   scaffolded config is silently untracked. Propose a check + warning in the
   skill.
3. Rules templates ship as literal TODO placeholders; a repo that runs the skill
   and stops inherits placeholder rules. Consider stack-aware default content.
4. Public-repo `write`-perms hardening guidance for the workflow assets.

## Execution sequencing

1. Preflight: confirm clean working tree state and that verification prereqs are
   still present.
2. Run the scaffold engine (`bash "$ENGINE" . python`); relay WRITTEN/SKIPPED.
3. Reconcile `.gitignore`, rules, workflows, `CLAUDE.md`, `pyproject.toml`.
4. Verify locally: `ruff check .`, `ruff format --check .`, `pytest`.
5. Land changes onto PR #12's branch; get it reviewed and merged to `main`.
6. Open the separate verification PR; confirm the `claude[bot]` comment.
7. File the four follow-ups on `claude-shared`.

## Out of scope

- Rewriting `CLAUDE.md` content beyond adding the shared-docs pointer.
- Changing runtime dependencies or `requires-python`.
- Hardening the workflow trigger (filed as follow-up #4).
- Standardizing other repos (`retina-gui` etc.) — separate pilots.
