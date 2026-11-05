# Decisions log

Why we chose what we chose. Newest at the bottom.

## 2026-11-05 · Dataset: Crunchbase October 2013 export

- **Chosen:** [datahoarder/crunchbase-october-2013](https://github.com/datahoarder/crunchbase-october-2013). Exported October 2013, before Crunchbase's licence change on 12 December 2013, so it is CC-BY.
- **Rejected:** `notpeter/crunchbase-data`. It is a 2015 export under CC-BY-NC, not the clean CC-BY licence.
- **Rejected:** `mtwilliams/crunchbase-in-2013`. It holds no data, only a script that needs a Crunchbase API key.
- **Rejected as the main source:** the Y Combinator directory. It has company profiles but no investor history, so there is nothing to learn investor taste from.
- **Data checks run before choosing (pass mark: a few hundred investors with 5+ investments):**
  - 52,870 investment rows, 139 exact duplicates (to be dropped and reported in Phase 1).
  - 40,978 unique investor to company pairs, 10,552 investors, 11,572 companies.
  - 1,528 investors with 5+ companies. **Pass.**
  - Every invested company is found in the companies file. **Pass.**
  - Company descriptions: **none.** Only a category label (present for 98% of companies). **Fail**, see next decision.

## 2026-11-05 · Version 1 uses labels, Version 2 adds text

- The chosen data has no company descriptions, only a category label.
- **Decision:** Version 1 uses category, city and round type as company features. TF-IDF and embeddings move to Version 2, when real descriptions arrive.
- **Cost:** cold start (brand new startups) is weaker in Version 1. It will still be tested honestly against the "most popular" baseline.

## 2026-11-05 · Languages: Python and TypeScript

- **Python:** backend, data, maths, models, API. NumPy is fast enough for about 41k deals.
- **TypeScript:** the Node.js frontend, to catch mistakes before the page runs.
- **Maybe later:** SQL, only if free Postgres is used for the demo (Phase 8).
- **Not used:** Rust, Go or C++ for speed. Speed is not the bottleneck, and it would make the project harder to explain.

## 2026-11-05 · Code licence: MIT

- MIT covers the code only. The data keeps its CC-BY licence and attribution.

## 2026-11-05 · Workflow

- Branches: `feature/...` to `staging` to `main`, every step through a PR with CI.
- The first skeleton commit goes straight to `main`, the only exception, because an empty repo has no branch yet.
- Files are created in the phase they belong to, never as empty placeholders.
- The scoreboard (Phase 2) is built before any model, so models are never judged by a scoreboard built after seeing their results.

## 2026-11-05 · Branch protection on `main` and `staging`

- Changes can only arrive through a pull request. No direct pushes.
- The `backend-tests` CI check must pass before a PR can merge.
- No bypassing, the owner included. Force pushes and branch deletion are blocked.
- **"Require approvals" is deliberately off.** GitHub does not let you approve your own PR, and this is a one-person project, so requiring approval would block every merge. Review still happens: the owner reads every PR before merging.
- **Proof:** this rule was tested with the `feature/phase-0-wrapup` PR into `staging`, then `staging` into `main`.
