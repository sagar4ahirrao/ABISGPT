# Security Policy — ABISGPT

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |
| < 0.1 | ❌ |

## Reporting a vulnerability

**Do not open a public GitHub issue for security reports.**

Email: **security@[replace-with-domain]** (PGP key to be published at
`.well-known/security.txt`). Include:

1. Affected component (`abisgpt.encoder`, `abisgpt.pad`, `abisgpt.index`,
   `abisgpt.pipeline`, `abisgpt.vault`, scripts, docs claims).
2. Reproduction steps / PoC. Synthetic data only — never attach real
   biometric samples, templates, or dumps.
3. Step-by-step impact assessment if achievable.

We aim to acknowledge reports within **72 hours** and publish advisories with
patches on a best-effort timeline. Credit is given unless you request
anonymity.

## Scope

In scope: this repository's code, reference configs, CI pipelines, and
published claims (e.g., the privacy properties asserted in the white paper —
"embeddings are not reversible" claims are security-relevant statements and
welcome for adversarial review).

Out of scope: downstream deployments, third-party cloud services (Qdrant
Cloud, Milvus operators), the Triton/TensorRT runtime itself, and social
engineering.

## Design-level security notes

- This is **research software**. Nothing here has completed a security audit;
  do not deploy against production PII.
- The "vector vault" (`abisgpt.vault`) is a reference for the decoupling
  pattern (raw media destroyed post-inference; UUID↔PII mapping lives in an
  isolated, independently encrypted relational store). It demonstrates the
  pattern with a local SQLite facade — swap in your own hardened store.
- Embedding inversion is an active research field. Treat "cannot be reversed"
  claims as probabilistic, not absolute; follow defenses such as
  embedding-space noise or secure-enclave inference if your threat model
  includes insiders with vector-db read access.

## Hardening checklist for anyone forking this for deployment

- [ ] mTLS between edge capture, inference, and vector tiers
- [ ] AES-256 at rest for vector DB volumes and any relational mirror
- [ ] Raw imagery deleted post-inference (stateless Triton workers only)
- [ ] UUID↔PII relational store behind separate creds + firewall
- [ ] Rate limiting + anomaly detection on `/search` and `/enroll`
- [ ] Model weights pinned + checksummed (supply-chain)
- [ ] Audit logs for every identification decision

