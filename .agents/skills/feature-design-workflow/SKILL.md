---
name: feature-design-workflow
description: Use when designing a non-trivial xLLM feature end-to-end, especially when the user wants a repeatable workflow from project context discovery and goal clarification through architecture, object/class design, implementation planning, validation, and review. This skill coordinates project-context-discovery, architecture-design, object-model-design, feature-development, and code-review instead of replacing them.
---

# Feature Design Workflow

Use this skill to keep feature design work ordered and reviewable. It is a workflow coordinator: load the specialized skills only when their phase is reached or when the user explicitly asks for that depth.

## Trigger

Use this skill when the user asks to:

- design a new feature or refactor workflow end-to-end
- decide whether roles, modules, classes, or boundaries are reasonable
- turn a rough proposal into architecture plus implementation plan
- define a repeatable design process for xLLM work
- split design thinking, architecture, object model, and implementation planning

Do not use this skill for small local bug fixes, simple code edits, or pure code reviews.

## Workflow

### Collaboration Rule

For non-trivial feature design, do not jump from analysis directly to a final design. Present tradeoffs and ask for confirmation at decision checkpoints before continuing to lower-level design or implementation planning.

Confirm with the user when:

- multiple viable architecture options exist
- a choice changes module ownership or public interfaces
- a choice affects compatibility, config, protocol, deployment, or runtime defaults
- a new abstraction or service boundary is introduced
- an assumption cannot be verified from local code
- the next phase would make the previous decision expensive to change

Use concise confirmations. Prefer one clear recommendation plus alternatives and consequences. Do not ask for confirmation on trivial implementation details.

### 1. Problem Framing

Purpose: make the target falsifiable.

Output:

- user-visible behavior
- success criteria
- non-goals
- hard constraints
- compatibility-sensitive surfaces

Use `architecture-design` when the goal, constraints, or system boundary need clarification.

### 2. Current-State Analysis

Purpose: anchor the design in the actual codebase.

Inspect:

- existing entry points
- module ownership
- lifecycle boundaries
- adjacent patterns
- constraints from `AGENTS.md` and required project style files

Output:

- project fact model
- current flow
- process and lifecycle model
- ownership map
- reusable local patterns
- modules that must remain untouched
- constraints that rule out obvious but wrong designs

Use `project-context-discovery` for this phase. For non-trivial xLLM runtime, distributed, serving, scheduler, model, config, protocol, or API changes, this phase is mandatory before architecture design.

Checkpoint:

- Confirm the current-state interpretation if it changes the initial user proposal.

### 3. Boundary Reasoning

Purpose: prove why each role should or should not exist.

For each candidate responsibility, answer:

- Why not put this in an existing object?
- What lifecycle does this responsibility belong to?
- What state does it own?
- What must it not know?
- What failure should it surface?

Output:

- accepted roles
- rejected alternatives
- rationale for each boundary

Checkpoint:

- Present 1 to 3 realistic options.
- State the recommended option.
- Explain tradeoffs in coupling, lifecycle, ownership, testability, rollout risk, and compatibility.
- Ask the user to confirm before finalizing the architecture when the choice is consequential.

Use `architecture-design` for this phase.

### 4. Architecture Design

Purpose: define the system shape.

Output:

- module ownership
- control flow
- data flow
- config and protocol impact
- state ownership
- failure handling and invariants
- compatibility, migration, rollout
- risks and open questions

Use the `architecture-design` output structure unless the user asks for a shorter form.

Checkpoint:

- Confirm the proposed architecture before object model design if it introduces new modules, services, protocol fields, config fields, or runtime defaults.

### 5. Object Model Design

Purpose: convert architecture roles into implementable objects without inventing shallow classes.

For each object or class, define:

- lifecycle
- owner
- owned state
- public interface
- construction path
- collaborators
- non-responsibilities
- failure behavior
- tests that prove the boundary

Rules:

- Do not create a class only to forward one call.
- Do not wrap stateless one-off functions in classes.
- Keep config input, immutable topology, request state, and process state separate.
- Introduce structured input/output objects only when they carry domain meaning, invariants, reuse, or meaningful signature simplification.

If object boundaries dominate the task, keep this as a separate document section or separate document after architecture.

Use `object-model-design` for this phase.

Checkpoint:

- Confirm major classes, ownership, and lifecycle choices before implementation planning.

### 6. Implementation Planning

Purpose: make the design executable in small, verifiable steps.

Output:

- ordered change list by module
- file or directory scope
- stage gates
- tests and validation commands
- compatibility notes
- unresolved questions that block coding

Use `feature-development` when the user wants implementation-ready steps or actual code changes.

Checkpoint:

- Confirm the staged rollout plan before coding if the change spans multiple modules or public behavior.

### 7. Review

Purpose: verify the implementation still matches the design.

Review for:

- role drift
- lifecycle mixing
- hidden coupling
- over-abstraction
- silent fallback
- weak validation
- missing failure tests
- config, API, protocol, or rollout regressions

Use `code-review` for implementation diffs or PR-style review.

## Recommended Documents

For large features, split outputs:

- `*_project_context.md`: phase 2
- `*_architecture_design.md`: phases 1-4
- `*_object_model_design.md`: phase 5
- `*_implementation_plan.md`: phase 6
- review comments or PR notes: phase 7

For smaller features, combine phases 1-6 into one document but keep sections distinct.

## xLLM Defaults

- Prefer fail-fast unless fallback is explicit, observable, testable, and semantically safe.
- Extend existing modules before adding parallel paths.
- Keep process-level entry, request-level orchestration, node service boundary, and engine-internal lifecycle separate.
- Tie abstractions to stable domain concepts, lifecycle boundaries, invariants, or real variation points.
- Surface unresolved questions instead of silently choosing risky assumptions.

## Completion Criteria

A design workflow is complete when:

- goals and non-goals are explicit
- project facts and current constraints are grounded in code
- process, lifecycle, and ownership models are explicit
- accepted and rejected boundaries are justified
- architecture names concrete modules and flows
- object model defines lifecycle and ownership
- implementation plan is ordered and testable
- risks and blocking questions are visible
