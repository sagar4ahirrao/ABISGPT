# Contributing to OpenABIS

Thanks for considering a contribution. This project publishes a *technical
white paper plus reference implementation*, so submissions fall into two
buckets — research writing and code — with a shared bar for evidence.

## Ground rules

1. **Signed commits are not required**, but every PR must contain a
   reproducible change: a failing test, a benchmark script, or a doc claim
   with citations.
2. **Never commit real biometric data.** Synthetic fixtures only
   (see `tests/fixtures`). Any PR detected to include personally identifiable
   raw samples will be rejected and reported.
3. **License:** MIT. You keep copyright on contributions; by submitting a PR
   you grant an unlimited, perpetual, royalty-free license to your
   contribution under MIT.
4. **Style:** `ruff` + `black` enforced by CI. Run:

   ```bash
   pip install -e ".[dev]"
   ruff check openabis scripts tests --fix
   black openabis scripts tests
   pytest -q
   ```

5. **Front-load the design.** For architectural changes (new backend, new PAD
   modality), open an issue with an RFC sketch first. Include the *failure
   modes* you considered — reviewers will push back on untested edge cases.

## Review anchors

Reviewers will look for:

- **Correctness:** does the test actually pin down the behavior?
- **Latency:** newBackend commands should degrade gracefully (not fire
  `ef_search=512` on every call).
- **Honesty:** do doc/README claims match what the code can reproduce on
  this commit? Any "99.9% recall" statement must link to a script + dataset
  you can share or justify.
- **Safety:** any new PAD/liveness sign reaffirms
  [`docs/RESPONSIBLE-USE.md`](docs/RESPONSIBLE-USE.md) boundaries.

## PR checklist

- [ ] Unit test added (`tests/`)
- [ ] `pytest -q` passes locally
- [ ] `ruff check` + `black --check` clean
- [ ] Docs updated (`docs/*.md` or README) if behavior/CLI changed
- [ ] No real biometric data in the diff
- [ ] CHANGELOG or release notes entry if user-facing

## Large contributions

For model weights, datasets, or >5k-line refactors, please discuss first in
the issue tracker. Model artifacts live in external storage (Hugging Face
Hub, Zenodo) and are referenced from the code — not committed to git.
