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

**Phase 3 merged:** PR #11 into `staging` (59/59 on Python 3.12), PR #12 into `main`.

**Phase 4 · Matrix factorisation (same session)**
- Denzel chose to build Phase 4 now and do a full walkthrough of everything from the beginning afterwards.
- Explained with "sliders" (hidden profiles) and the rookie with no game tape (new startups get profiles from their labels).
- Found and fixed an averaging bug before testing (nudges 256 times too small).
- Matrix factorisation alone: tie overall, much better on known, worse on new. Not promoted.
- Denzel chose option B: a hybrid. Weight 0.8 chosen on the practice split. 0.0203 vs 0.0171, but not proven (CI crosses zero). Not promoted.
- Decision: label similarity stays the starter. Both challengers stay in the code on the bench. No more tweaking on this test set.

**Next**
- Phase 4 PR.
- The full walkthrough from the beginning (Denzel's request), including cosine similarity, matrix factorisation and gradients. Probably as a guide PDF part 2.
- Then Phase 5: Thompson sampling.

**Phase 4 merged:** PR #13 into `staging` (79 passed, 3 expected failures on Python 3.12), PR #14 into `main`.

**Walkthrough, option C (PDF, then chat)**
- Delivered the maths guide Part 2 (14 pages, 11 diagrams): a one-page recap of Part 1, checklists (vectors), cosine similarity, the leak check, sliders (matrix factorisation), BPR, sigmoid and loss, gradients and gradient descent with one step by hand, the gradient check, the averaging bug, the practice split, the hybrid and z-scores, and why nothing was promoted.
- Next: go through it together in chat, one small piece at a time.

## 2026-11-16 · Session 5: Phase 5

- Denzel gave the assistant's name a meaning: M.A.R.A. = Machine-learning Automation and Recommendation Agent (the recommendation engine plus the CI/CD for models).
- Walkthrough of guide Part 2 postponed again by Denzel ("we'll do it later"). Still planned.
- Denzel asked where PHP could fit. Answer: it does not earn a spot (Python backend, TypeScript frontend already cover it). Denzel let it pass.
- Exposure check first: the starter is already fair to new startups, so Thompson sampling was given the job of fixing the hybrid instead (Denzel chose option B after asking whether skipping pieces is right).
- Thompson sampling on the hybrid: highest NDCG so far (0.0215), new startups' share 22% to 64%, but two near misses. Not promoted, recorded.
- Denzel asked if there was something better to do. Answer: strengthen the scoreboard with several test periods. Denzel agreed.

**Next**
- Phase 5 PR, then the multi-season scoreboard (Phase 5b).

**Phase 5 merged:** PR #15 into `staging` (90 passed, 5 expected failures on Python 3.12), PR #16 into `main`.

**Phase 5b · Multi-season scoreboard (same session)**
- Counted test investors per 12-month season before writing pass marks: 2009 467, 2010 524, 2011 623, 2012 732. Data ends Oct 2013.
- Denzel chose: retune per season (the old practice year would leak into the 2011 season) and four seasons (2009 to 2012).
- Built: a test window end for the split and the draft board, a movable practice cutoff for tuning, a cluster bootstrap in the maths, and `evaluation/seasons.py`.
- Result: Thompson on the hybrid vs label similarity +0.0012 over all seasons, +0.0023 on the fresh seasons; both could be luck. Not promoted, recorded. 112 passed, 7 expected failures locally.
- Slip: MARA ran `git status` on Denzel's Mac (against the no-git rule). Checked straight away: no lock file, nothing changed.

**Next**
- Phase 5b PR (`feature/phase-5b-multi-season` into `staging`), then promote to `main`.
- Then Phase 6 (explain each pick), or the postponed walkthrough of guide Part 2.

**Phase 5b merged:** PR #17 into `staging` (112 passed, 7 expected failures in 6 minutes on CI), PR #18 into `main`.

**Phase 6 · Explain each pick (same session)**
- Started the guide Part 2 walkthrough (cosine similarity worked example), then Denzel chose to finish the build first and do the maths step by step afterwards.
- Plan agreed: explain only the starter, exactly (category part + state part = score).
- Built `explain/label_explanations.py` and `explain/report.py`. All 5 pass marks met on 8,280 real picks.
- Finding: startups missing a label are 2% of the board but 36% of the picks. Logged, not fixed in this phase.
- 127 passed, 7 expected failures locally.

**Next**
- Phase 6 PR (`feature/phase-6-explain-picks` into `staging`), then promote to `main`.
- Decide: test a fix for the missing-label bias, or go on to Phase 7 (quality gate).
- After the build: the maths walkthrough, step by step (Denzel's request).

**Phase 6 merged:** PR #19 into `staging` (127 passed, 7 expected failures on CI), PR #20 into `main`.

## 2026-11-19 · Session 6: Phase 7, the quality gate

- Denzel chose relative fairness (option A) and required explainers ("A and yes") after asking which was best.
- Built `gate/` (registry, rules, promotion check, command line, release check), `champion.json` (label similarity, v1), a `quality-gate` CI job and a release workflow.
- The gate blocks matrix factorisation and the hybrid on all three rules, and Thompson on the hybrid on rules 1 and 3. A planted cheat passes rule 1, so the gate can say yes.
- 145 passed, 7 expected failures locally.

**Next**
- Phase 7 PR, then promote to `main`: this should create the first release, `model-v1`.
- Denzel to add `quality-gate` as a required check on `staging` and `main` (GitHub settings).
- Then the missing-label fix as the gate's first real challenger.
- After the build: the maths walkthrough, step by step.
