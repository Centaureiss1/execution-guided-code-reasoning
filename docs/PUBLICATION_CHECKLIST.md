# Public release checklist

The portfolio is organized for review, but publishing it is a separate decision. Resolve the items below before pushing to a public host.

## Required owner decisions

- [ ] **Choose a license.** No license is included because reuse terms must be selected by the repository owner.
- [ ] **Confirm dataset redistribution rights.** Review the licenses and terms for OpenCodeInstruct, Magicoder, KodCode, HumanEval, MBPP/EvalPlus, and LiveCodeBench before publishing derived or copied task data.
- [ ] **Confirm generated-output policy.** Decide whether full prompts, completions, tests, and failure traces should be public or whether only summaries and small samples should be hosted.
- [ ] **Confirm author metadata.** Verify the author name in `README*` and `CITATION.cff`; add contact, personal site, affiliation, or course details only if desired.
- [ ] **Choose the public repository URL and visibility.** Add the final URL to the README and citation metadata after creation.

## Repository-size strategy

The snapshot is approximately 443 MB. The largest individual files are about 56 MB, below GitHub's 100 MB hard per-file limit but above its 50 MB warning threshold.

Choose one approach:

- [ ] keep the full evidence archive in normal Git and accept a heavy clone;
- [ ] track large JSON/JSONL/log artifacts with Git LFS;
- [ ] publish source and summaries in Git, and attach the full archive as a versioned release asset or external research artifact;
- [ ] publish a small representative data sample and document how authorized users can regenerate the rest.

The current folder keeps all artifacts locally so this choice remains reversible.

## Privacy and secret review

- [ ] Run a secret scanner on the final Git history, not only the working tree.
- [ ] Review console logs for hostnames, usernames, internal endpoints, scheduler IDs, or environment dumps.
- [ ] Review generated samples for unintentionally retained source metadata.
- [ ] Remove stale `.pid` files if operational provenance is not valuable to public readers.
- [ ] Verify that no model-provider API keys or `.env` files are staged.

The initial organization pass found configuration placeholders and environment-variable names, but no embedded credential. It also found machine-specific filesystem paths in the original scripts and logs; those are documented as portability limitations.

## Presentation polish

- [ ] Add a repository description and topics such as `llm-post-training`, `grpo`, `code-generation`, `react-agent`, `evalplus`, and `livecodebench`.
- [ ] Add one architecture/result image to the repository social preview if visual branding is useful.
- [ ] Replace the text-only author section with verified links, if desired.
- [ ] Decide whether the original proposal PDF should remain public.
- [ ] Create a tagged release for the preserved 2026-05-04 experiment snapshot.

## Optional engineering work

- [ ] Replace absolute paths with a shared configuration layer.
- [ ] Add pinned dependency files or a container image.
- [ ] Add a small, CPU-safe smoke test that does not download models.
- [ ] Add continuous integration for the `react-bench` unit tests.
- [ ] Generate compact result tables automatically from evaluator JSON files.
- [ ] Add a hardened execution backend before running untrusted model code.
