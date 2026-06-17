# Multi-Agent Git/GitHub Sync Guide

This repository may be edited by three agents:

- Agent A1: Codex on computer A
- Agent A2: ClaudeCode in VSCode on computer A
- Agent B1: Codex on computer B

The goal is to keep all work synchronized through GitHub without overwriting each other's changes.

## GitHub Account & Repository

| Item | Value |
|---|---|
| GitHub owner (account) | `Henry-WYF` |
| Repository | `MAC-VO_test` |
| Clone URL (HTTPS) | `https://github.com/Henry-WYF/MAC-VO_test.git` |

All three agents push to the **same** repository `Henry-WYF/MAC-VO_test`. Agents are
distinguished by **branch name**, not by repository.

> Note: not every working copy is a git repository yet. For example, Agent A2's VSCode copy on
> computer A currently reports `git remote -v` → "not a git repository" and has no remote. The
> repository only needs to be created and pushed **once** (see "Repository Bootstrap"); after
> that, every machine attaches to the same `origin`. The human decides which agent performs the
> one-time bootstrap and when each commit is pushed.

## Core Rules

1. Use Git and GitHub as the single source of truth.
2. Never edit directly on `main` for feature work.
3. Each agent works on its own branch.
4. Pull before starting work.
5. Commit before switching tasks or handing work to another agent.
6. Push every meaningful checkpoint to GitHub.
7. Do not use destructive commands unless the human explicitly approves them.

Forbidden unless explicitly approved:

```bash
git reset --hard
git checkout -- .
git clean -fd
git push --force
```

## Branch Naming

Use one branch per task and include the agent name. Each agent has a fixed branch prefix:

| Agent | Prefix |
|---|---|
| A1 (Codex, computer A)      | `agent-a-codex/` |
| A2 (ClaudeCode VSCode, comp A) | `agent-a-claude/` |
| B1 (Codex, computer B)      | `agent-b-codex/` |

Recommended names:

```text
agent-a-codex/global-pgo-fix
agent-a-claude/loop-closure
agent-b-codex/review-global-pgo
```

For paper stages:

```text
feature/global-pgo
feature/loop-closure
feature/cov-aware-loop
feature/imu-tight-coupling
```

If multiple agents work on the same stage, still split by agent:

```text
agent-a-codex/loop-place-recognition
agent-a-claude/loop-geometric-verification
agent-b-codex/loop-review-tests
```

## Repository Bootstrap (one time, performed once by whichever agent the human designates)

The remote repository is created and seeded **exactly once**. Run this only in a working copy
that already contains the intended initial source, and only after confirming `origin/main` does
not already exist on GitHub (otherwise skip to "Initial Setup On Each Computer"):

```bash
git init                       # if the working copy is not a repo yet
git add -A
git commit -m "Initial unmodified MAC-VO source"   # see commit policy below
git branch -M main
git remote add origin https://github.com/Henry-WYF/MAC-VO_test.git
git push -u origin main
```

After this push succeeds, every other machine **joins** the existing remote via "Initial Setup"
below — they must not repeat the bootstrap or push a competing `main`.

## Initial Setup On Each Computer

If the repository is not cloned yet:

```bash
git clone https://github.com/Henry-WYF/MAC-VO_test.git
cd MAC-VO_test
```

If the repository already exists locally:

```bash
git remote -v
git fetch origin
git status
```

If a local working copy exists but has no remote yet (e.g. an uninitialized VSCode copy) **and
`origin/main` already exists on GitHub**, do not push it as a new `main` — that would clobber
the existing history. Instead attach to the existing remote and reconcile:

```bash
git init                       # if not a repo yet
git remote add origin https://github.com/Henry-WYF/MAC-VO_test.git
git fetch origin
# Inspect differences before integrating; never force-push over existing history.
git status
git checkout -b agent-name/task-name origin/main   # base new work on the remote main
```

The repository path is `Henry-WYF/MAC-VO_test`.

## Identity Configuration (each agent, each computer)

Set the git author on every machine:

```bash
git config user.name  "Henry-WYF"
git config user.email "<the email registered on the Henry-WYF GitHub account>"
```

All three agents authenticate as the same GitHub account (`Henry-WYF`); they are told apart by
branch name and commit message, not by author identity. Because the author is identical
everywhere, the branch prefix (`agent-a-claude/…` etc.) is the **only** reliable signal of which
agent produced a commit — always use it.

## Start A New Task

Always start from the latest `main`:

```bash
git checkout main
git pull --ff-only origin main
git checkout -b agent-name/task-name
```

Example:

```bash
git checkout main
git pull --ff-only origin main
git checkout -b agent-b-codex/review-global-pgo
```

## Save Work

Before committing, inspect changes:

```bash
git status
git diff
```

Stage only relevant files:

```bash
git add path/to/file1.py path/to/file2.yaml
git commit -m "Implement concise task summary"
```

Push the branch:

```bash
git push -u origin agent-name/task-name
```

Commit message style:

```text
Implement GlobalPGO edge registration
Fix GlobalPGO loop edge validation
Add synthetic GlobalPGO tests
Document loop closure integration plan
```

Avoid vague messages:

```text
update
fix
changes
misc
```

## Sync While Working

If `main` changed while an agent is working:

```bash
git fetch origin
git checkout agent-name/task-name
git merge origin/main
```

If there are conflicts:

1. Stop and inspect conflicted files.
2. Resolve conflicts manually.
3. Run targeted tests if possible.
4. Commit the merge.

```bash
git status
git add resolved/file.py
git commit --no-edit
git push
```

Do not discard another agent's code to make a conflict disappear.

> Always pass `-m "..."` (or `--no-edit` for a merge) on commits. A bare `git commit` opens an
> interactive editor, which hangs an automated agent.

## Pull Request Workflow

Use GitHub Pull Requests to merge feature branches into `main`.

Each PR should include:

```text
Summary:
- What changed
- Why it changed

Validation:
- Tests run
- Tests not run and why

Risk:
- Possible behavior changes
- Files needing extra review
```

Recommended PR merge order for this project:

1. GlobalPGO fixes and tests
2. Loop closure interface and place recognition
3. Loop geometric verification
4. Loop edge insertion into GlobalPGO
5. Evaluation scripts and paper experiment configs
6. Optional covariance-aware loop weighting
7. Optional IMU module

## Stable Version Policy

`main` should always be runnable or at least statically coherent.

Before merging into `main`, verify at minimum:

```bash
pytest Scripts/UnitTest/test_global_pgo.py -q
pytest Scripts/UnitTest/test_config_macvo.py -q
```

If the local machine cannot run tests because dependencies are missing, write that in the PR:

```text
Not run locally: pytest/pypose/CUDA environment unavailable.
Needs server validation.
```

Do not modify CUDA, TensorRT, FlowFormer, model loading, or dependency versions just to make local tests pass.

## Large Files And Data

Large artifacts must stay out of git. The repo's `.gitignore` already covers the main offenders
(`Model/`, `*.pth`, `*.pkl`, `Results/`, `exp/`, `Temp/`, `cache/`, `wandb/`). Rely on
`.gitignore` rather than manual discipline:

```text
Model/
*.pth
*.pkl
Results/
exp/
Temp/
cache/
```

Before the one-time bootstrap, run `git status` and confirm none of the above are staged — the
bootstrap uses `git add -A`, which would otherwise commit them. If something large is missing
from `.gitignore`, add it there first.

Heads-up: `.gitignore` also ignores `*.json` and `*.png` globally. Test PNG assets are
re-included via `!Scripts/UnitTest/assets/**/*.png`. If you add a config/data file with one of
those extensions that *should* be tracked, add a matching `!`-exception or it will be silently
skipped.

Use GitHub only for code, configs, tests, and small documentation files.

## Handoff Message Template

When one agent finishes a work session, leave this message for the others:

```text
Branch:
Commit:

Changed files:
- ...

What was done:
- ...

Validation:
- ...

Known issues:
- ...

Next suggested step:
- ...
```

## Current Project Context

Current intended development route:

1. Keep the original MAC-VO behavior intact.
2. Maintain the initial unmodified source as the first Git commit.
3. Keep GlobalPGO as an optional module controlled by config.
4. Add loop closure next.
5. Use GlobalPGO as the backend that consumes loop edges.
6. Consider covariance-aware loop weighting after basic loop closure works.
7. Consider IMU tight coupling only after the VO + GlobalPGO + loop closure system is stable.

Baseline commits expected on `main` (in order). Identify them by message, not by hash — the
short SHAs differ per environment and change if the repository is re-initialized:

```text
Initial unmodified MAC-VO source
Implement Phase 1 GlobalPGO
```

## GlobalPGO Configuration Rule

GlobalPGO must remain optional.

Old configs without `Odometry.global_pgo` must behave like the original MAC-VO.

Enabled config example:

```yaml
Odometry:
  optimizer:
    type: TwoFrame_PGO
    args:
      device: cpu
      vectorize: true
      parallel: true
      graph_type: disp
      autodiff: false

  global_pgo:
    enabled: true
    optimize_on_terminate: true
    max_iterations: 50
    trans_weight: 100.0
    rot_weight: 100.0
    device: cpu
    include_interp_frames: false
```

Disabled behavior:

```yaml
global_pgo:
  enabled: false
```

or no `global_pgo` block at all.

## Quick Commands

Check current state:

```bash
git status
git log --oneline --decorate --graph -n 10
```

Create branch:

```bash
git checkout main
git pull --ff-only origin main
git checkout -b agent-name/task-name
```

Commit:

```bash
git add <files>
git commit -m "Clear summary"
```

Push:

```bash
git push -u origin agent-name/task-name
```

Update branch with latest main:

```bash
git fetch origin
git checkout agent-name/task-name
git merge origin/main
```

