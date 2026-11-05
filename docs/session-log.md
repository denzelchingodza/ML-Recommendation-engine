# Session log

What each working session did, what was decided, and where we stopped.
Newest session at the bottom. Read this first at the start of every session.

## How we work (standing agreements)

- One task at a time. Say which phase and step we are on.
- Talk ideas through and say what will change before changing it.
- Pass marks are written down **before** any run. Failures are reported honestly.
- Branches: `feature/...` into `staging`, then `staging` into `main`. Both are protected: PR plus green CI, no bypassing.
- Always check a feature PR's base is `staging`.
- Denzel runs `git commit` and `git push` on his Mac. Mara writes and tests the code, and opens and merges PRs when asked.
- Nothing is pushed until it is proven to work.
- Public data only. Never Funema or MAN.T.EL data, never Harvest code.
- No em dashes anywhere. No tables in chat. Basketball analogies. Diagrams where possible.
- Before any maths file is merged, walk through it line by line (agreed from Phase 2 on).

## 2026-11-05 · Session 1 (morning): planning, Phase 0 and Phase 1

**Planning**
- Drew the full system diagram and the roadmap (Phases 0 to 9).
- Data: Crunchbase October 2013 export (CC-BY), from `datahoarder/crunchbase-october-2013`. Rejected the 2015 CC-BY-NC copy and an empty mirror.
- The data has no company descriptions, only category labels. Decision: Version 1 uses labels, Version 2 adds text.
- Languages: Python backend, TypeScript frontend. Licence: MIT.

**Phase 0 · Setup: done**
- Created the public repo `denzelchingodza/ML-Recommendation-engine`, CI on every PR, branch protection on `main` and `staging`.
- Mistake: PR #1 was merged into `main` instead of `staging`. Synced `staging` in PR #2, corrected the decision log in PR #3, proved the full flow with PR #4.

**Phase 1 · Data: done (PR #5, #6)**
- Data committed with SHA-256 fingerprints. Mac Roman encoding.
- Cleaned: 139 duplicates, 2 junk rows, 1 undated deal dropped. 40,977 investor to company pairs.
- Split at 1 Jan 2012 (single cutoff, no future leakage): 28,614 training pairs, 5,878 hidden test pairs, 828 test investors.
- One pass mark label corrected after looking at the real rows (logged, no code changed).
- Finding: 73% of hidden test deals are brand new companies.

## 2026-11-05 · Session 2 (afternoon): Phase 2

**Phase 2 · Scoreboard: done (this PR)**
- Taught by hand first: recall@k (Denzel mixed it up with precision, then got it), NDCG (DCG and normalising), bootstrap confidence intervals (the 20 shots vs 1,000 shots example).
- Built: `maths/ranking_metrics.py`, `maths/confidence_intervals.py`, `models/baselines.py`, `evaluation/draft_board.py`, `evaluation/scoreboard.py`.
- Draft board: 4,693 companies raising on or after the cutoff. Known catch logged.
- All 5 pass marks met. 47 tests pass.
- Results: random NDCG@10 0.0020, most popular 0.0055. Most popular scores exactly 0 on new companies.
- The line-by-line walkthrough of the maths files was postponed: Denzel asked for a PDF explaining the maths, the code and its history instead.

**Where we stopped**
- Merging Phase 2, then writing the PDF (maths explained simply with examples and diagrams, the backstory of each piece of maths, then the code and how it fits the project), covering Phases 0 to 2.

**Next**
- Phase 3 · Find candidates: company label vectors and cosine similarity, aimed at the new-company gap.
