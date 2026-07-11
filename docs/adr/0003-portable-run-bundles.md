# ADR-0003: Portable and Atomic Run Bundles

Status: accepted

## Context

The repository has multiple manifest schemas. Artifact paths depend on the process working directory, and a Git SHA may not identify uncommitted source.

## Decision

One versioned RunManifest will describe every Run. Artifact references are relative to the run root or use an explicit durable URI. Every required artifact has a content hash.

The source identity includes the commit SHA, tracked diff hash, untracked source hash, and an explicit clean or dirty state. Dataset and model revisions must reflect resolved inputs, not requested labels.

Run finalization is atomic: persist evidence, validate hashes and semantics, write the final manifest, apply promotion, then update campaign state. An interrupted transaction may resume idempotently.

## Consequences

- Run bundles remain valid after relocation.
- Reports and lifecycle commands use one RunBundle reader.
- A non-clean source tree remains allowed but cannot masquerade as the recorded commit.
