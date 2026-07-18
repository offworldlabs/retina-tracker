# setup-repo Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the Offworld Labs standard repo setup on `retina-tracker` by running `core:setup-repo`, reconciling the files it skips, and proving a `claude[bot]` review comment posts on a PR.

**Architecture:** Extend the existing open PR #12 (branch `fix/claude-review-comments`, which already fixed `claude-code-review.yml`) with the rest of the standardization. The scaffold engine writes the 6 missing files; the remaining work is hand-reconciling the skipped files (`.gitignore`, rules, `claude.yml`, `pyproject.toml`, `CLAUDE.md`). Merge to `main`, then verify with a separate non-workflow PR (the `claude-code-action` default-branch guard forbids verifying on the workflow-fix PR itself).

**Tech Stack:** Bash scaffold engine (`core:setup-repo`), GitHub Actions, `uv`/`ruff`/`pytest`, `gh` CLI.

## Global Constraints

- **Never clobber a pre-existing file.** The engine guarantees this; hand-edits must preserve existing content.
- **Keep `requires-python = ">=3.10"`** in `pyproject.toml` (tracker runs on ARM/edge). Only `ruff target-version` moves to `py312`.
- **Keep the standard `pull_request` trigger** on workflows (public repo; hardening is a follow-up, not done here).
- **All work lands on branch `fix/claude-review-comments`** (PR #12), not directly on `main`.
- Ruff config: `line-length = 120`, `select = ["E", "F", "W"]`, `quote-style = "double"`.
- Verification prereqs already satisfied: `CLAUDE_CODE_OAUTH_TOKEN` secret exists, Claude GitHub App installed, default branch is `main`.

---

### Task 1: Check out PR #12's branch and preflight

**Files:** none (git + inspection only)

**Interfaces:**
- Produces: a clean working tree on branch `fix/claude-review-comments`, verified even with `main`.

- [ ] **Step 1: Confirm clean working tree, then check out the branch**

```bash
cd /Users/jonnyspicer/repos/retina/retina-tracker
git status --porcelain        # expect: only untracked *.md/*.sh scratch files from before; no staged changes
git fetch origin fix/claude-review-comments
git checkout fix/claude-review-comments
git pull --ff-only origin fix/claude-review-comments
```

- [ ] **Step 2: Verify branch state matches the plan's assumptions**

```bash
git rev-list --left-right --count origin/main...HEAD    # expect: 0<TAB>1 (even with main, 1 ahead)
grep -nE "pull-requests:|issues:" .github/workflows/claude-code-review.yml   # expect: write / write
grep -nE "pull-requests:|issues:" .github/workflows/claude.yml               # expect: read / read (fixed in Task 6)
```

Expected: review workflow already `write`; `claude.yml` still `read`.

- [ ] **Step 3: Confirm verification prerequisites are still present**

```bash
gh secret list | grep CLAUDE_CODE_OAUTH_TOKEN     # expect: one row
gh repo view --json visibility --jq .visibility   # expect: PUBLIC (informs the follow-up, not a blocker)
```

No commit (inspection only).

---

### Task 2: Run the scaffold engine

**Files:**
- Create (by engine): `.claude/settings.json`, `.claude/rules/security.md`, `.claude/rules/code-style.md`, `.editorconfig`, `.github/workflows/ci.yml`, `tests/.gitkeep`

**Interfaces:**
- Consumes: branch from Task 1.
- Produces: the 6 scaffolded files on disk (not yet committed).

- [ ] **Step 1: Run the engine for the python stack**

```bash
ENGINE="${CLAUDE_PLUGIN_ROOT:-/Users/jonnyspicer/.claude/plugins/marketplaces/offworld/plugins/core}/skills/setup-repo/scripts/scaffold-repo.sh"
bash "$ENGINE" . python
```

- [ ] **Step 2: Verify WRITTEN / SKIPPED output**

Expected `WRITTEN`: `.claude/settings.json`, `.claude/rules/security.md`, `.claude/rules/code-style.md`, `.editorconfig`, `.github/workflows/ci.yml`, `tests/.gitkeep`.
Expected `SKIPPED`: `CLAUDE.md`, `.github/workflows/claude-code-review.yml`, `.github/workflows/claude.yml`, `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`.

If any of the 6 above lands in SKIPPED instead, stop — the branch already had it and the plan's assumptions need rechecking.

- [ ] **Step 3: Confirm settings.json content is the org standard**

```bash
cat .claude/settings.json    # expect: extraKnownMarketplaces.offworld -> offworldlabs/claude-shared, enabledPlugins."core@offworld": true
```

No commit yet (the `.claude/*` files cannot be `git add`ed until Task 3 fixes `.gitignore`).

---

### Task 3: Un-ignore the shared `.claude/` config in `.gitignore`

**Files:**
- Modify: `.gitignore` (the `# Claude` section)

**Interfaces:**
- Consumes: scaffolded `.claude/` files from Task 2.
- Produces: a `.gitignore` that tracks `.claude/settings.json` and `.claude/rules/` while still ignoring `.claude/settings.local.json`.

- [ ] **Step 1: Replace the `.claude/` blanket ignore**

Change the existing two-line section:

```
# Claude
.claude/
```

to:

```
# Claude (track shared config; keep personal/local settings ignored)
.claude/*
!.claude/settings.json
!.claude/rules/
```

- [ ] **Step 2: Verify git now tracks shared config but not local settings**

```bash
git check-ignore -v .claude/settings.json        # expect: NO output (not ignored)
git check-ignore -v .claude/rules/security.md     # expect: NO output (not ignored)
git check-ignore -v .claude/settings.local.json   # expect: matched by .claude/* (still ignored)
git status --short .claude/                        # expect: settings.json + rules/*, NOT settings.local.json
```

- [ ] **Step 3: Commit the gitignore fix plus the scaffolded config**

```bash
git add .gitignore .claude/settings.json .claude/rules/ .editorconfig .github/workflows/ci.yml tests/.gitkeep
git commit -m "chore: scaffold org-standard Claude config, CI, and editorconfig

Runs core:setup-repo (python stack) and un-ignores .claude shared config so
settings.json and rules/ are tracked while settings.local.json stays personal.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Fill the rules files with real content

**Files:**
- Modify: `.claude/rules/code-style.md`, `.claude/rules/security.md`

**Interfaces:**
- Consumes: placeholder rules from Task 2 (already committed in Task 3).
- Produces: rules reflecting this repo's actual conventions (sourced from `CLAUDE.md` "Code Style Guidelines").

- [ ] **Step 1: Replace `.claude/rules/code-style.md` entirely**

```markdown
<!--
  Shared code-style rules for Offworld Labs repos, adapted for retina-tracker.
-->

# Code Style Rules

- Write minimal, self-documenting code; prefer clarity over cleverness.
- Do not add comments; let names and structure carry the intent.
- Match the existing patterns and conventions of the file you are editing.
- Cover all business logic with tests before marking work complete.
- Code must pass `ruff check` (E, F, W) and `ruff format --check` at line-length 120.
```

- [ ] **Step 2: Replace `.claude/rules/security.md` entirely**

```markdown
<!--
  Shared security rules for Offworld Labs repos, adapted for retina-tracker.
-->

# Security Rules

- Never commit secrets, credentials, or radar/node URLs — use environment variables or `.env` (git-ignored).
- Validate and sanitise external detection and ADS-B input at node/trust boundaries before it reaches the tracker.
- Keep dependencies patched and pinned; review new dependencies before adding them.
```

- [ ] **Step 3: Verify no TODO placeholders remain**

```bash
grep -rn "TODO" .claude/rules/     # expect: NO output
```

- [ ] **Step 4: Commit**

```bash
git add .claude/rules/
git commit -m "docs: fill shared Claude rules with retina-tracker conventions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Flip `claude.yml` permissions to write

**Files:**
- Modify: `.github/workflows/claude.yml` (permissions block, ~lines 22-24)

**Interfaces:**
- Consumes: `claude.yml` with `read` perms (unchanged by PR #12).
- Produces: `claude.yml` granting `pull-requests: write` / `issues: write`.

- [ ] **Step 1: Edit the permissions block**

Change:

```yaml
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write
      actions: read # Required for Claude to read CI results on PRs
```

to:

```yaml
    permissions:
      contents: read
      pull-requests: write
      issues: write
      id-token: write
      actions: read # Required for Claude to read CI results on PRs
```

- [ ] **Step 2: Verify**

```bash
grep -nE "pull-requests:|issues:" .github/workflows/claude.yml   # expect: write / write
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/claude.yml
git commit -m "fix(ci): grant @claude workflow write perms so responses post

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Align ruff target-version to py312 (keep requires-python 3.10)

**Files:**
- Modify: `pyproject.toml` (`[tool.ruff]` block)

**Interfaces:**
- Consumes: existing `pyproject.toml` with `target-version = "py310"`.
- Produces: `target-version = "py312"` matching CI's Python; `requires-python` untouched.

- [ ] **Step 1: Edit only the ruff target-version line**

In the `[tool.ruff]` block change:

```toml
[tool.ruff]
line-length = 120
target-version = "py310"
```

to:

```toml
[tool.ruff]
line-length = 120
target-version = "py312"
```

- [ ] **Step 2: Verify requires-python is unchanged and ruff still passes**

```bash
grep -n 'requires-python' pyproject.toml    # expect: requires-python = ">=3.10"  (UNCHANGED)
grep -n 'target-version' pyproject.toml     # expect: "py312"
ruff check .                                 # expect: no new errors
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: set ruff target-version to py312 to match CI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Add the Org-Wide Context pointer to `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (append one section at end; file is 161 lines, well under the 200 ceiling)

**Interfaces:**
- Consumes: existing `CLAUDE.md`.
- Produces: `CLAUDE.md` with a shared-docs pointer, without altering existing content.

- [ ] **Step 1: Append the section to the end of `CLAUDE.md`**

```markdown

## Org-Wide Context

For shared architecture, cross-service contracts, decisions, and runbooks, see
the Offworld Labs shared docs: https://github.com/offworldlabs/claude-shared/tree/main/docs

Shared Claude rules live in `.claude/rules/` and are enforced on every change.
```

- [ ] **Step 2: Verify the file is still under 200 lines and existing content is intact**

```bash
wc -l CLAUDE.md                       # expect: ~167, < 200
grep -n "Org-Wide Context" CLAUDE.md  # expect: one match near end
grep -n "System Overview" CLAUDE.md   # expect: original first section still present
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: link CLAUDE.md to shared org docs and .claude/rules

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Full local verification and push

**Files:** none (verification + push)

**Interfaces:**
- Consumes: all commits from Tasks 3-7.
- Produces: PR #12 updated on the remote with the full standardization.

- [ ] **Step 1: Run the mandatory verification commands (per repo CLAUDE.md)**

```bash
ruff check retina_tracker/ tests/ --select E,F,W --line-length 120
ruff format --check retina_tracker/ tests/
pytest tests/ -v
```

Expected: all three pass. If `ruff format --check` flags files you did **not** touch, that is pre-existing drift — do not fix it here; note it and proceed (it is out of scope per the spec).

- [ ] **Step 2: Push the branch (updates PR #12)**

```bash
git push origin fix/claude-review-comments
```

- [ ] **Step 3: Update the PR #12 title/body to reflect expanded scope**

```bash
gh pr edit 12 --title "Adopt core:setup-repo standard setup (issue #5)" --body "$(cat <<'EOF'
Pilots `core:setup-repo` on retina-tracker (offworldlabs/claude-shared#5).

## What changed
- Scaffolds org-standard `.claude/settings.json`, `.claude/rules/*`, `.editorconfig`, uv-based `ci.yml`.
- Un-ignores shared `.claude/` config in `.gitignore` (keeps `settings.local.json` personal).
- Fills rules with retina-tracker conventions.
- Fixes **both** Claude workflows to `write` perms; review workflow also gets `track_progress` + `--allowedTools` (write perms alone did not post comments).
- Aligns ruff `target-version` to py312 (keeps `requires-python >=3.10`).
- Links CLAUDE.md to shared org docs.

## Verification
The `claude-code-action` default-branch guard means the review comment cannot be
verified on this PR. After merge to `main`, a separate non-workflow PR confirms a
`claude[bot]` comment posts.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Confirm the PR is green and mergeable**

```bash
gh pr view 12 --json mergeable,mergeStateStatus,statusCheckRollup --jq '{mergeable,mergeStateStatus}'
```

Expected: `MERGEABLE` / `CLEAN` (or `BLOCKED` only on required-review, which is the human gate below).

---

### Task 9: Merge PR #12 to main (human review gate)

**Files:** none (GitHub merge)

**Interfaces:**
- Consumes: green PR #12.
- Produces: standardization on `main`; workflows now correct on the default branch.

- [ ] **Step 1: Request review / obtain approval**

This is a human gate — do not self-merge without the repo owner's approval. Confirm with the owner, then:

```bash
gh pr merge 12 --squash --delete-branch
```

- [ ] **Step 2: Verify main now has write perms on both workflows**

```bash
git fetch origin main
git show origin/main:.github/workflows/claude-code-review.yml | grep -E "pull-requests:|issues:"   # write / write
git show origin/main:.github/workflows/claude.yml              | grep -E "pull-requests:|issues:"   # write / write
```

No commit.

---

### Task 10: Verify a `claude[bot]` comment posts (separate non-workflow PR)

**Files:**
- Create: a trivial change on a new branch that touches **no** `.github/workflows/` file (e.g. a one-line clarification in `CLAUDE.md` or `docs/`).

**Interfaces:**
- Consumes: fixed workflows on `main` from Task 9.
- Produces: evidence (a `claude[bot]` review comment + a zero-denial run log) satisfying issue #5's acceptance criteria.

- [ ] **Step 1: Create a tiny verification PR**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b verify/claude-review-comment
# make a trivial, non-workflow edit, e.g. append a blank line to docs/superpowers/plans/2026-07-17-setup-repo-pilot.md
git commit -am "test: trivial change to verify Claude review comment posts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push origin verify/claude-review-comment
gh pr create --title "Verify Claude review comment posts" --body "Non-workflow PR to confirm the corrected review workflow posts a claude[bot] comment (issue #5 acceptance)." --base main
```

- [ ] **Step 2: Wait for the review run and confirm a comment posted**

```bash
sleep 60
gh pr view --json comments --jq '.comments[].author.login' | grep -i claude   # expect: claude bot login present
```

If no comment after a few minutes, read the run log for the denial:

```bash
RUN=$(gh run list --workflow=claude-code-review.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$RUN" --log | grep -iE 'permission_denials_count|Workflow validation failed'
```

Expected on success: `permission_denials_count: 0` and no validation-failure line.

- [ ] **Step 3: Close/clean up the verification PR**

```bash
gh pr close verify/claude-review-comment --delete-branch
```

No commit to main from this task.

---

### Task 11: File skill-friction follow-ups on claude-shared

**Files:** none (GitHub issues)

**Interfaces:**
- Consumes: findings from Tasks 2-10.
- Produces: tracked follow-ups referencing issue #5, satisfying its "file any friction" criterion.

- [ ] **Step 1: File the four follow-up issues**

```bash
gh issue create --repo offworldlabs/claude-shared \
  --title "setup-repo: core claude-code-review.yml lacks track_progress/allowedTools" \
  --body "Piloting in retina-tracker (#5): the bare core review workflow grants write perms but omits \`track_progress: true\` and \`--allowedTools \"mcp__github_inline_comment__create_inline_comment,...\"\`. Without them the code-review plugin's inline-comment tool is denied and no comment posts even with write scope. Propose upstreaming these into assets/ci/claude-code-review.yml. Refs #5."

gh issue create --repo offworldlabs/claude-shared \
  --title "setup-repo: detect .gitignore that ignores .claude/" \
  --body "Piloting in retina-tracker (#5): the repo's .gitignore had a blanket \`.claude/\` rule, so scaffolded settings.json + rules/ would never be committed. The skill should detect this and warn / offer to un-ignore shared config while keeping settings.local.json ignored. Refs #5."

gh issue create --repo offworldlabs/claude-shared \
  --title "setup-repo: rules templates ship as TODO placeholders" \
  --body "Piloting in retina-tracker (#5): assets/rules/{security,code-style}.md are literal TODOs. A repo that runs the skill and stops inherits placeholder rules. Consider stack-aware default content or a post-scaffold prompt to fill them. Refs #5."

gh issue create --repo offworldlabs/claude-shared \
  --title "setup-repo: public-repo hardening guidance for write-perm workflows" \
  --body "Piloting in retina-tracker (#5, a PUBLIC repo): pull-requests: write on a pull_request trigger is fork-exploitable. The assets/runbook should document when to switch to pull_request_target + author allow-list for public repos. Refs #5."
```

- [ ] **Step 2: Verify the issues were created**

```bash
gh issue list --repo offworldlabs/claude-shared --search "setup-repo in:title" --json number,title
```

- [ ] **Step 3: Comment on issue #5 with the pilot outcome**

```bash
gh issue comment 5 --repo offworldlabs/claude-shared --body "Pilot complete on retina-tracker (PR #12 merged). Standard .claude/, workflows, and tooling landed with no clobbered files; a claude[bot] review comment verified on a separate non-workflow PR. Four friction follow-ups filed."
```

No commit.

---

## Self-Review

**Spec coverage:** Every spec section maps to a task — scaffold writes (T2), `.gitignore` (T3), rules (T4), `claude.yml` perms (T5), `pyproject` (T6), `CLAUDE.md` pointer (T7), the review-workflow crux (already on the branch, verified in T1), verification sequence (T9-T10), the four follow-ups (T11). `claude-code-review.yml` needs no task because PR #12 already reconciled it — verified in Task 1 Step 2.

**Placeholder scan:** No TBD/TODO steps; every edit shows exact before/after content and exact commands with expected output. The only "TODO" mention is the grep in Task 4 Step 3 confirming placeholders are *gone*.

**Type/name consistency:** Branch name `fix/claude-review-comments`, workflow filenames, and the `--allowedTools` string are used identically across tasks. Verification commands reference the real workflow file name `claude-code-review.yml` throughout.
