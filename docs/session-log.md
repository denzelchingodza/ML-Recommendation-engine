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
- Mara never runs git commands on Denzel's Mac (git needs to delete its own lock files there). Mara only reads and writes project files; Denzel runs git.

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

**Wrap-up**
- Phase 2 merged: PR #7 into `staging` (47/47 tests on Python 3.12), PR #8 into `main`.
- Delivered the maths guide PDF (12 pages, 8 diagrams): the story, the idea, a worked example and the code for the cutoff split, recall, NDCG, the bootstrap and the baselines.
- Mistake: Mara ran `git fetch` on Denzel's Mac, which left `.git/objects/maintenance.lock` behind. It was renamed to `.git/stale-maintenance-lock-from-mara` (safe to delete). New standing rule above: Mara never runs git on the Mac.

**Where we stopped**
- End of day. This log update still needs its own small PR (`feature/session-log-nov-5` into `staging`, then `main`).

**Next**
- Phase 3 · Find candidates: company label vectors and cosine similarity, aimed at the new-company gap.

## 2026-11-06 · Session 3: log PR, Phase 3 started

**Done**
- Merged the 5 Nov session log: PR #9 into `staging`, PR #10 into `main`.
- Local `staging` was out of date, so `git checkout staging` refused to switch. Fixed with `git checkout staging` and `git pull`. New habit: run those two at the start of every session.

**Phase 3 · Find candidates: planning**
- Checked the labels on the 4,693 draft board companies: category (41 kinds, 1% missing) and state (47 kinds, 2% missing) are usable.
- Dropped: city (712 values, too thin), country (every company is in the USA, so this is a US-only dataset), round type (belongs to deals, and new companies only have rounds after the cutoff, so it would leak the future).
- Proposal: Phase 3 uses category + state. Not yet confirmed by Denzel.
- Teaching: a company as a vector, explained as a checklist (tick = 1, empty = 0). Denzel found the number version confusing; the checklist version landed better.

**Where we stopped**
- Mid-exercise: reading a filled-in checklist backwards (software ticked, NY ticked: answer is a software company in NY). Pick up here.

**Side quest: frontend design sketch (not part of the repo)**
- Denzel asked for a standalone HTML sketch of what the demo could look like.
- v1 (mint, airy, "why" colour bar, fit score): Denzel said it felt AI-made, too airy, no actions, hard to understand.
- v2: dense deal-flow list (stage, raised, last round, location), one plain sentence for "why", actions (Add to pipeline, Pass, Note) with Pipeline and Passed tabs and Undo, filters (stage, first-time raisers).
- v3: same as v2 in military green (olive #556B2F, dark olive ink). Denzel liked it.
- Takeaway for Phase 8: lead with the facts investors check first, give real actions, explain each pick in one plain sentence. The "why" must come from the real signals behind the score (Phase 6).

**Next**
- Finish vectors, then cosine similarity by hand, then confirm category + state, then pass marks, then code.

**Weak spots discussion (6 Nov, after the sketch)**
- Listed the project's limits for David: predicts what investors back, not what succeeds; 2013 US-only data; thin labels; only established investors tested; "didn't invest" is not "didn't like"; one test period; not live.

## 2026-11-09 · Session 4: Phase 3 built

- Denzel asked to build first and understand the vectors along the way. The checklist exercise is answered (software ticked, NY ticked = software company in NY = [1, 0, 0, 0, 1]); cosine similarity has not been taught by hand yet.
- Confirmed: labels are category + state. Pass marks written before the run.
- Built `maths/cosine_similarity.py` and `models/label_similarity.py`, added the model to the scoreboard.
- All 6 pass marks met. 59 tests pass (with the leak check).
- Results: NDCG@10 overall 0.0171 (3 times most popular), new companies 0.0172 (most popular: 0), known companies a tie.
- Leak check: shuffled labels collapse to 0.0005. Kept as a permanent test.

**Next**
- Phase 3 PR, then a short walkthrough of cosine similarity (and a guide PDF section) when Denzel wants it.
- Phase 4: matrix factorisation with hand-written gradients, combined with the labels for new companies.
