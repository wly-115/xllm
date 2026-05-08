---
name: object-model-design
description: Use after architecture boundaries are mostly decided and before implementation, when designing concrete classes, structs, interfaces, ownership, lifecycles, state placement, construction flow, and abstractions for xLLM features. Emphasizes project coding standards, DDD-style domain concepts, fail-fast behavior, and avoiding shallow or unnecessary abstractions.
---

# Object Model Design

Use this skill to translate architecture roles into implementable objects without over-designing. The output should make coding straightforward and reviewable.

## Required Context

Before designing objects under `xllm/`, read the project custom code style:

- `.agents/skills/code-review/references/custom-code-style.md`

If the object model is part of a larger feature design, use this after `architecture-design` and before `feature-development`.

## Workflow

### Collaboration Rule

Object model design must show reasoning, not only final class definitions. When a design choice creates a new abstraction, owns state, introduces polymorphism, changes lifecycle, or affects module boundaries, present the tradeoff and ask the user to confirm before moving to implementation planning.

For each consequential choice, include:

- option considered
- why it works
- why it is risky
- recommended choice
- reason this choice is easier to test and maintain

Do not ask for confirmation on local naming or mechanical style choices.

### 1. Restate Architecture Roles

List the roles already accepted by the architecture design.

For each role, capture:

- purpose
- owning module
- lifecycle category
- collaborators
- non-responsibilities

Lifecycle categories should be explicit:

- process-level
- initialization-time
- immutable runtime topology
- request-scoped runtime state
- node-internal runtime state
- transport/client state

### 2. Identify Domain Objects

Only introduce an object when it maps to a stable domain concept, lifecycle boundary, invariant, ownership boundary, or real variation point.

Good reasons to introduce an object:

- it owns state with invariants
- it represents a durable domain concept
- it separates different lifecycles
- it hides a replaceable strategy
- it prevents cross-module coupling
- it gives a small stable interface over complex behavior

Bad reasons to introduce an object:

- it only forwards one call
- it wraps a stateless one-off function
- it exists only to make a class diagram look complete
- it stores fields that do not share a domain meaning
- it hides unclear responsibilities behind a long name
- it creates a parallel abstraction over an existing good local pattern

Checkpoint:

- Before accepting new classes or interfaces, explain why existing objects, free functions, or plain structs are insufficient.
- Ask for confirmation if the new abstraction becomes a module boundary or public extension point.

### 3. Place State Deliberately

Separate these state categories:

- config input: parsed external declaration
- derived immutable topology: indexes, dependency graph, final outputs
- process state: services, engines, ports, gflags-derived options
- request state: inputs, intermediate values, node execution states
- transport state: channels, clients, connection metadata

Rules:

- Do not put request state in immutable graph objects.
- Do not put engine lifecycle state in graph config objects.
- Do not put config parsing behavior in runtime execution objects.
- Do not let clients own server or engine lifecycle.
- Do not let adapters manage transport connections.

Checkpoint:

- If state could reasonably live in more than one object, compare the options by lifecycle, ownership, mutation frequency, and testability.

### 4. Define Ownership and Construction

For each class or struct, define:

- who constructs it
- who owns it
- whether ownership is unique or shared
- when it is destroyed
- whether it is immutable after construction
- whether it can be copied or moved
- what must be initialized before it

Prefer:

- `std::unique_ptr` for sole ownership
- `std::shared_ptr` only for genuine shared lifetime
- references or raw pointers only for non-owning relationships with clear lifetime
- explicit constructors for single-argument construction
- `final` for classes not designed for inheritance

Checkpoint:

- Ask for confirmation before choosing shared ownership, global state, singleton access, or cross-module lifetime coupling.

### 5. Design Minimal Interfaces

For each public method, define:

- caller
- input contract
- output contract
- failure behavior
- side effects
- thread-safety expectation if relevant

Keep interfaces small:

- avoid exposing mutable containers directly
- avoid methods that mix construction, validation, execution, and cleanup
- avoid broad `Context` objects unless they represent a real request or lifecycle boundary
- prefer free functions for stateless build/validate helpers
- use virtual interfaces only for real polymorphic variation points

Checkpoint:

- For every virtual interface or registry, state the real variation point and why a simpler concrete type is not enough.

### 6. Choose Struct vs Class

Use `struct` only for plain data aggregation with no member functions.

Use `class` when the type:

- enforces invariants
- owns resources
- has behavior
- controls mutability
- hides implementation details

Do not add methods to structs in xLLM code.

Tradeoff rule:

- Prefer `struct` for plain config or parsed data.
- Prefer `class` when invariants, ownership, or controlled mutation matter.
- Prefer free functions when behavior is stateless and does not need ownership.

### 7. Define Failure Semantics

xLLM defaults to fail-fast.

For each object, specify:

- which invalid states are impossible by construction
- which invalid inputs return `false` / status
- which unrecoverable programmer/config errors use `CHECK` or `LOG(FATAL)`
- which failures must include logs with identifying context
- which fallback paths are explicitly supported

Avoid:

- broad exception handling
- silent default values for invalid config
- probing multiple paths until one works
- falling back to semantically different behavior without observability

Checkpoint:

- Ask for confirmation before adding fallback behavior. Fallback must be explicit, observable, testable, and semantically safe.

### 8. Validate the Object Model

Check the design with these questions:

- Can each class be explained in one sentence?
- Does each class have one lifecycle category?
- Does each field belong to that lifecycle?
- Is every mutable field necessary?
- Could a free function be clearer?
- Is any class just forwarding?
- Are ownership and destruction obvious?
- Can tests prove each invariant?
- Does this reuse existing xLLM patterns before adding new ones?

## Output Format

Use this structure unless the user asks for a different format:

1. Scope and assumptions
2. Architecture roles being refined
3. Object inventory
4. Lifecycle and ownership
5. Class and struct responsibilities
6. Public interfaces
7. State placement
8. Construction and destruction flow
9. Failure semantics and invariants
10. Tradeoffs and rejected abstractions
11. User confirmation checkpoints
12. Test plan
13. Open questions

## xLLM Style Requirements

Apply project-specific style when the design will touch `xllm/`:

- C++ class / struct names use `PascalCase`.
- Functions use `snake_case`.
- Member variables use trailing underscore.
- New C++ files require project copyright headers.
- Classes not intended for inheritance should be `final`.
- Single-argument constructors should be `explicit`.
- Prefer fixed-width integers.
- Avoid `auto` for primitive/simple types.
- Use `CHECK` / `LOG(FATAL)` for assertions or unrecoverable errors according to local style.
- Use project-root-relative includes.

## Completion Criteria

An object model design is complete when:

- every accepted object has a clear lifecycle and owner
- every field has an explicit state category
- public interfaces are minimal and tied to callers
- invalid states and failures are defined
- shallow abstractions are rejected explicitly
- consequential tradeoffs are documented
- required user confirmations are resolved or listed as blockers
- tests can verify the main invariants and collaboration paths
