# Agent workflow

`AGENTS.md` at the repository root is the operational authority. Follow it even
when an older document or global skill suggests launching training directly.

## Work narrowly and verify claims

- State assumptions and distinguish source-code behavior, test evidence, and
  unmeasured hypotheses. Historical LLM notes are not proof.
- Inspect current code and the actual resolved run config before changing a plan.
- Prefer the smallest change that addresses a demonstrated problem. Do not add
  speculative heads, losses, configuration switches, or unrelated refactors.
- Preserve other contributors' edits. Use focused changes and verification;
  do not reset a dirty tree or mix experiments into fixes.
- Report exact commands, inputs, outputs, results and skipped tests. A green
  test suite does not prove untested production combinations are correct.

## Execution boundaries

Training, training-style validation and benchmark workloads are user-started via
Make. Agents may offer commands, then must ask whether the user ran them and wait
for confirmation before monitoring. Never launch, stop, resume, or background
those workloads autonomously. Small focused unit tests and read-only inspection
are allowed. Data audits/conversion require an explicit execution request and
recorded inputs, output directory, log and result; derived data stays outside Git.

Status requires the host Python-worker identity, output/log paths, GPU mapping,
and progress across observations. A sandbox process list or an old PID is not
proof that a job stopped. Keep launcher and Python-worker PIDs separate.

## Documentation and skills

Start with [the documentation index](../README.md) and
[the training guide](../training/TRAINING_GUIDE.md).
Keep substantive project docs under `Docs/`, in English; root/tool entrypoints
may be short discovery pointers. Run records, migration notes and superseded evidence stay local, outside Git.

All MIND skills have one maintained source in `.agents/skills/`:

- [`validate-feature`](../../.agents/skills/validate-feature/SKILL.md)
- [`strict-contracts-and-validation`](../../.agents/skills/strict-contracts-and-validation/SKILL.md)

| Tool | Repository rules | Skill discovery |
|---|---|---|
| Codex | `AGENTS.md` | `.agents/skills/` |
| OpenCode | `AGENTS.md` | `.agents/skills/` |
| Claude Code | `CLAUDE.md` imports `@AGENTS.md` | `.claude/skills/<name>` symlinks to `.agents/skills/<name>` |

There is no single documented directory scanned directly by all three tools.
The two relative Claude symlinks provide compatibility without duplicate skill
content. Git preserves these links on Linux; they do not depend on Anzu paths.
Do not create another project copy under `.codex`, `.opencode`, or Docs.

After cloning, start the tools inside this repository with project trust/settings
enabled. Confirm both skills are listed: Codex `/skills`, Claude's skill menu or
`/validate-feature`, and OpenCode's available skill list. In Claude, `/context`
should show `CLAUDE.md` and its imported repository instructions. Restart after
adding a previously absent skill root. Invocation loads instructions; these
skills do not independently launch training or require Python package installs.

Check the local paths directly:

```bash
test -f .agents/skills/validate-feature/SKILL.md
test -f .agents/skills/strict-contracts-and-validation/SKILL.md
test -f .claude/skills/validate-feature/SKILL.md
test -f .claude/skills/strict-contracts-and-validation/SKILL.md
readlink .claude/skills/validate-feature
readlink .claude/skills/strict-contracts-and-validation
```

Personal/plugin skill snapshots remain separate from these repo skills. Do not
reactivate the stale global `validate-feature` copy when restoring Anzu history.
Keep its archived source if needed; use the repo version for MIND.

Discovery and symlink behavior are documented by
[Codex](https://learn.chatgpt.com/docs/build-skills),
[OpenCode](https://opencode.ai/docs/skills/),
[Claude skills](https://code.claude.com/docs/en/skills), and
[Claude's AGENTS.md import](https://code.claude.com/docs/en/memory#agentsmd).
Paths were checked locally with Codex CLI 0.140.0 and OpenCode 1.18.30;
Claude Code was not installed on Anzu during this review, so target discovery
must still be checked in the actual Claude client.

Generated data, checkpoints, private tool state and logs must not be added to
Git. Project documentation and shared skills are versioned and arrive with a
clone; private history, datasets, logs and outputs need migration separately.
