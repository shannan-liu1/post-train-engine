# Repository Agent Instructions

Treat the user as an expert. Write in active subject-verb-object sentences and remain precise and concise. Optimize for truth, simplicity, and verified behavior. Treat assumptions as conjectures: inspect evidence, surface conflicts, and challenge weak plans.

Make the smallest coherent change that solves the real problem. Preserve behavior unless the user explicitly requests change. Design for simplicity, cohesion, low coupling, and restrained reuse. Prefer one canonical path. Avoid speculative architecture, dependencies, configuration, and indirection. Fail loudly with actionable errors.

Report exactly what changed, where, why, how it was tested, what stayed unchanged, and remaining risks. Never claim validation that you did not perform. Do not use the em dash when writing.

## Agent skills

- Use `CONTEXT.md` for domain vocabulary and invariants.
- Use `docs/specs/canonical-research-flywheel.md` as the governing product and architecture contract.
- Use `docs/plans/canonical-research-flywheel-implementation.md` for execution order, file ownership, and verification commands.
- Read `docs/runpod_deployment.md` before creating, updating, or connecting to a RunPod Pod.
- Read applicable decisions under `docs/adr/` before changing a canonical interface.
- Track repository work in local Markdown until the user explicitly selects an external issue tracker.
- Keep exactly one mutable execution path. Compatibility commands must delegate to that path.
- Keep promotion deterministic and fail closed. No training or compute adapter may promote a candidate.
- Update the spec progress ledger and unknown-unknown register after each verified slice.
