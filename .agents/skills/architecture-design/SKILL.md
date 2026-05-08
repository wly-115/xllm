---
name: architecture-design
description: Draft or review technical designs, architecture proposals, module boundaries, and implementation approaches before coding or during refactors. Use when Codex needs to clarify goals, non-goals, constraints, ownership boundaries, design options, tradeoffs, migration impact, risks, rollout, or validation for a feature, subsystem change, integration, or cross-module design.
---

# Architecture Design

Use this skill to turn vague implementation ideas into reviewable, testable, and scoped technical plans. Prefer concrete boundaries, explicit tradeoffs, and verifiable success criteria over generic architecture language.

## Workflow

1. Restate the problem in terms of user-visible behavior, constraints, and success criteria.
2. Inspect the existing code path, ownership boundaries, and adjacent local patterns before proposing new structure.
3. Define goals, non-goals, assumptions, and constraints explicitly. Do not silently fill gaps.
4. Identify the minimum coherent design that solves the problem. Prefer extending existing modules over adding parallel paths.
5. If the design is not obvious, present 1 to 3 realistic options, explain tradeoffs, and recommend one.
6. Specify the chosen design in implementation terms: owning modules, APIs, protocol fields, config, state flow, error behavior, and rollout.
7. End with focused validation, risks, and open questions.

## Design Rules

- Prefer the smallest coherent design that satisfies the goal.
- Tie abstractions to stable domain concepts, real variation points, lifecycle boundaries, or invariants.
- Do not invent configurability, fallback paths, or extension points that are not required.
- Preserve fail-fast behavior unless fallback is an explicit, observable, testable part of the design.
- Separate current problems, proposed mechanism, and validation plan. Do not mix them.
- Surface ambiguity, missing information, or conflicting constraints instead of guessing.
- If an existing local pattern is good enough, reuse it and say why.

## Output Contract

Use this structure unless the user asks for a different format:

1. Background
2. Goals
3. Non-goals
4. Constraints and assumptions
5. Current state
6. Options considered
7. Proposed design
8. Module boundaries and ownership
9. Data flow or control flow
10. Failure handling and invariants
11. Compatibility, migration, and rollout
12. Validation plan
13. Risks and open questions

Keep each section concrete. Name specific modules, interfaces, and behaviors instead of writing generic platform language.

## Review Mode

When the input is an existing plan, design doc, or proposal:

- Test whether the problem statement is complete and falsifiable.
- Check whether the proposed boundaries match existing ownership and layering.
- Look for unnecessary abstractions, hidden coupling, or unowned state.
- Look for compatibility risks in public APIs, config, protocols, persistence, and runtime defaults.
- Check whether validation is strong enough to catch regressions before rollout.
- Report the highest-risk issues first, then give the recommended adjustment.

## xLLM Guidance

For xLLM work, anchor the design in the actual directory ownership model:

- `xllm/api_service`, `xllm/server`: external service surfaces, request handling, serving behavior.
- `xllm/core/runtime`, `xllm/core/framework`, `xllm/core/scheduler`: execution orchestration, workers, scheduling, batching, and runtime state.
- `xllm/core/distributed_runtime`: distributed execution and PD serving behavior.
- `xllm/models`, `xllm/processors`, `xllm/parser`, `xllm/function_call`: model-specific behavior, preprocessing, parsing, and tool-call logic.
- `xllm/proto`, `xllm/c_api`, `xllm/cc_api`, `xllm/pybind`: boundary interfaces and compatibility-sensitive surfaces.

When drafting a design, name which layer owns each change and which layers must remain untouched.

## Implementation Handoff

If the user wants the design to be implementation-ready, finish with:

- Ordered change list by module
- Required tests or targeted validation commands
- Compatibility notes for config, API, protocol, or rollout
- Explicit unresolved questions that block coding
