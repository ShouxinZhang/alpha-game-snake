# Repo Agent Instructions (Generic)

## Language and Communication

- The user communicates in Chinese; by default, write documentation, logs, and explanations in Chinese.
- Communicate from a business-outcome perspective: clarify value, benefits, risks, and impact scope.

## Core Engineering Rules

- Priority order: business outcomes > long-term architecture > concise style.
- Read existing code before implementation; avoid redundant work.
- Keep changes minimal; avoid unauthorized scope expansion.
- Keep module boundaries clean; avoid cross-module pollution.
- New business code should be placed in leaf directories.

## Mandatory Skill Workflow

For each development task, follow `local-dev-workflow` and trigger sub-skills by condition:

- `build-check`: run quality gates after code changes.
- `dev-logs`: record each development cycle.
- `repo-structure-sync`: run after file/dir/dependency/script structural changes.
- `git-management`: run at milestones or before large changes.
- `domain-data-update`: run when modifying project-specific core domain data.

If the user explicitly names a skill (for example `$build-check`), prioritize it.

## Human Intent Alignment

Before any code change, align and confirm:

- files to modify/add
- core implementation approach
- impact scope and risks

Start implementation only after clear user confirmation.

## Required Pre-Change Checks

- Read `AGENTS.md` before each change.
- Prefer MCP tools for module-level context; use architecture docs as fallback.
- Create a backup before deletion/rollback operations.

## Quality and Verification

- Delivery without verification is prohibited.
- Run quality gates after changes:

```bash
bash scripts/check_errors.sh
```

- Run tests as needed:

```bash
npm test
```

## Documentation and Traceability

- Keep architecture docs synchronized with actual repository structure.
- Add a log under `docs/dev_logs/{YYYY-MM-DD}/` for each development cycle.
- Each log must include: user prompt, timestamp (seconds), file list, change details, and verification results.

## Automation and Version Policy

- Automate all repeatable steps whenever possible.
- Prefer latest verifiable stable versions for SDKs/dependencies.

## Prohibited Behaviors

- Committing without checks/tests.
- Finishing development without dev log update.
- Not syncing architecture docs after structural changes.
- Asking users to do scriptable work manually.
