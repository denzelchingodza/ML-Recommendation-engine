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
- **What actually happened:** PR #1 (`feature/phase-0-wrapup`) was merged straight into `main` by mistake, because the PR base was left on `main`. CI still had to pass, so the protection rules held. `staging` was then synced from `main` in PR #2.
- **Proof of the full flow:** the `feature/fix-decision-log` PR (this correction) goes into `staging`, then `staging` into `main`, with CI green at each step.
- **Lesson:** when opening a feature PR, always check the base branch is `staging`.

## 2026-11-05 · Phase 1 pass mark 2 corrected (no code changed)

- The pass mark originally said "3 undated rows dropped".
- The real-data test failed: the code dropped **2 rows with missing ids** and **1 undated row**.
- Looking at the actual rows: 2 are broken junk rows (one has the street address "57 Woodstock Road" where a company id should be, the other is empty), most likely from a line in the source file that broke apart. 1 is a real deal (Rockley Ventures into Kotura) with no date.
- The earlier quick check counted all 3 as "no date" because the junk rows also have no date. The code is more precise, so the pass mark was wrong, not the code.
- **Corrected pass mark:** 2 rows with missing ids and 1 undated row. The total of 3 dropped rows and the 40,977 pairs did not change.
- **No code was changed to make the test pass.** Changing a pass mark after a run is only acceptable when the real rows show the original count was mislabelled.

## 2026-11-05 · Phase 1: data is committed, with fingerprints

- The two CSVs (about 13 MB) are committed in `backend/data/raw/`, copied unchanged from the source at commit `9303ebdc`.
- **Why:** if the source mirror ever disappears or changes, the project can still be rebuilt, and CI never needs the internet for data.
- The loader checks each file's SHA-256 fingerprint and stops with a clear error if anything differs.
- **Rejected:** a download script, because the project would depend on someone else's repo staying online and unchanged.

## 2026-11-05 · Phase 1: one cutoff date, 1 January 2012

- Deals before the cutoff train the model. Deals on or after it are hidden for testing.
- **Rejected:** a random split. It leaks the future: a company's later fame (for example many investors in later rounds) would help "predict" earlier deals.
- **Rejected:** hiding each investor's latest 20%. It still leaks across investors.
- Cutoffs compared before choosing: Jul 2011, Jan 2012, Jul 2012, Jan 2013. Jan 2012 gave the most test investors (828) with a median of 4 hidden deals each, while keeping 28,614 training pairs.
- Only investors with 5+ training companies are tested. This leaves out 4,312 newer investors, so results describe established investors.
- Repeat investments in the same company keep the FIRST date, so a company first backed before 2012 can never appear in the test.

## 2026-11-05 · Phase 1 finding: 73% of test deals are brand new companies

- 4,275 of the 5,878 hidden test pairs involve a company with no deals before 2012.
- **Why it matters:** matrix factorisation (Phase 4) learns one vector per company from history, so it cannot score a company it has never seen. Company labels must carry these cases.
- **Consequence for Phase 2:** the scoreboard must report brand new and known companies separately, or results will be misread.

## 2026-11-05 · Phase 1: Mac Roman encoding

- The CSVs are not UTF-8. Read as Mac Roman, names like "MiaSolé" come out correctly.
- A few names were already garbled in the source (for example `agÃ¡mi Systems`). They are left unchanged rather than guessed.

## 2026-11-05 · Phase 2: the draft board

- Each investor's ranking covers the **4,693 companies that raised money on or after 1 Jan 2012** (1,552 known, 3,141 new), minus companies that investor already backed before the cutoff.
- **Rejected:** ranking all 17,727 companies. Most stopped raising years before 2012, so an investor could not back them, and the scores would be meaningless.
- **Known catch:** the board quietly reveals which companies survived long enough to raise money after the cutoff. It does not reveal who invested, which is the actual answer. The catch is the same for every model, so comparisons stay fair.
- Every hidden test company is checked to be on the board. If one is missing, scoring stops with an error.

## 2026-11-05 · Phase 2: scoring rules

- **k = 10:** an investor glancing at a top-10 list.
- **Ties are broken randomly** with a fixed seed, never alphabetically, so no company gets a free boost from its name.
- **Segments:** overall, known companies and new companies. An investor with no hidden deals in a segment is left out of that segment (NaN), never counted as a 0.
- **Confidence intervals:** 95% percentile bootstrap over investors, 10,000 resamples. Model comparisons use a **paired** bootstrap (each investor compared with themselves).
- scikit-learn and scipy are used **only in tests**, to check the hand-written maths.

## 2026-11-05 · Phase 2 finding: most popular cannot see new companies

- Most popular beats random overall (NDCG@10 +0.0035, CI +0.0007 to +0.0066) and on known companies (+0.0107).
- On new companies it scores exactly 0 and loses to random (-0.0018, CI -0.0035 to -0.0005). Known companies always have higher popularity, so its top 10 never contains a new company.
- **Consequence:** with 73% of hidden deals being new companies, the next model must use company labels to score new companies at all.
- **Caution for the PR gate:** hits are rare, so intervals are wide compared with the scores. Small improvements will often read as "could be luck".

## 2026-11-09 · Phase 3: labels are category and state

- All 4,693 draft board companies are in the company table. Category: 41 values, 1% missing. State: 47 values, 2% missing.
- **Rejected:** city (712 values, too thin), country (every company is in the USA, so this is a US-only dataset), round type (it belongs to deals, and new companies only have rounds after the cutoff, so it would leak the future).
- A missing label means no tick. A company with no labels scores 0 instead of breaking the maths.
- Category and state get equal weight. Not tuned.

## 2026-11-09 · Phase 3: label similarity results

- Investor checklist = sum of the checklists of the companies they backed before the cutoff (each company once). Score = cosine similarity.
- NDCG@10 overall 0.0171 vs most popular 0.0055 vs random 0.0020. New companies 0.0172 vs 0 for most popular.
- On known companies it ties with most popular (-0.0024, CI -0.0102 to +0.0055). The prediction before the run was that most popular would win there; that was wrong.
- **Leak check (written before running):** labels shuffled between companies must collapse the score to around random. Result: 0.0005 (CI 0.0000 to 0.0011), overlapping random. Kept as a permanent test. Shuffled is slightly below random because wrong labels point the model at the wrong companies.

## 2026-11-09 · Phase 4: matrix factorisation design

- Startup profile = sum of its label profiles (category, state) + its own personal profile if it had deals in training. New startups still get a profile from their labels.
- Trained with the pairwise ranking loss (BPR, Rendle et al. 2009): "the investor backed A, not a random B", nudging profiles so A scores higher. Gradients written by hand, checked against numerical gradients (worst relative error about 1e-8).
- **Bug found before testing:** the loss was first averaged over each batch of 256, so every nudge was 256 times too small and the model stayed at coin-flip level (loss 0.698 to 0.693). Switched to summing over the batch, the standard for BPR. Loss then fell from 0.68 to 0.18 over 30 epochs.
- **Settings chosen on a practice split inside the training years** (learn before 2011, check on 2011), never on the hidden 2012 test. Grid written before the run: dim 16 or 32, reg 0.01 or 0.001. Chosen: dim 32, reg 0.01. Tuning refuses data from the test period.

## 2026-11-09 · Phase 4 result: matrix factorisation not promoted

- NDCG@10 overall 0.0172 vs label similarity 0.0171: a tie (CI -0.0052 to +0.0054).
- Known startups: much better (0.0207 vs 0.0088, real win). New startups: worse (0.0119 vs 0.0172, real loss).
- Pass marks 7 (beat label similarity overall) and 8 (not clearly worse on new startups) failed. Kept as strict expected-to-fail tests so the result stays visible.
- Leak check passed: training on scrambled history collapses the score to around random.

## 2026-11-09 · Phase 4: hybrid, and why we stopped there

- The two models are strong in opposite places, so a hybrid was tried: w x standardised matrix factorisation + (1 - w) x standardised label similarity.
- **Honesty note:** this idea came AFTER seeing the matrix factorisation test results. To limit that, w was chosen only on the practice split (w = 0.8 from 0.0 to 1.0 in steps of 0.1) and the pass marks were written before the hybrid run.
- Result: overall 0.0203 vs 0.0171, but the paired CI (-0.0016 to +0.0080) crosses zero. The win is not proven, so it was not promoted. Not clearly worse on new startups (vs label similarity) or known startups (vs matrix factorisation).
- **Decision:** label similarity stays the starter. Matrix factorisation and the hybrid stay in the code, tested, on the bench. No further tweaking on this test set: the test results have now been seen twice, and tweaking until something passes is how projects fool themselves.
- A bench model can be promoted later with new evidence (for example Version 2 data or a new test period) through the Phase 7 quality gate.

## 2026-11-16 · Phase 5: exposure check before Thompson sampling

- Checked whether the problem Thompson sampling fixes exists before building it. New startups are 67% of the board. Share of top-10 slots: most popular 0% (the same 15 startups for everyone), label similarity 72%, matrix factorisation 27%, hybrid 22%.
- The starter is already fair, so Thompson sampling alone would add complexity for nothing. Denzel asked whether skipping pieces is right; the answer was to give it a real job instead: make the hybrid fair.
- The exposure check is now reported for every model and should become a Phase 7 gate rule: a new starter must not starve new startups.

## 2026-11-16 · Phase 5: Thompson sampling, simplified

- Score = standardised base score + wobble, wobble ~ Normal(0, strength / sqrt(1 + deals)). Classic Thompson sampling needs live feedback, which we do not have; this keeps its idea (more uncertainty, more exploration) and is labelled as simplified in the code.
- Strength grid and selection rule written before the run: on the practice split, among strengths where new startups get at least their board share, pick the best NDCG@10. Chosen: 0.25 (practice: 69% new share).
- Results on the hidden test: NDCG@10 0.0215 (highest so far). Against the hybrid: real win on new startups (+0.0070), real loss on known (-0.0097). New startups' share 64%, different startups shown 2,869.
- Pass marks 5 (fairness, 64% vs 67%) and 7 (beat label similarity, CI -0.0002 to +0.0094) failed. Recorded as strict expected-to-fail tests. **Not promoted.**
- **Not tweaked after seeing the test:** a bigger strength might have reached 67%, but choosing it now would be test-peeking.

## 2026-11-16 · Next: a multi-season scoreboard

- Two near misses in a row (hybrid, Thompson sampling) show the scoreboard's limit: one test period with 828 investors can only prove gains above about 0.005.
- Plan: repeat the test at several cutoffs (for example 2010, 2011, 2012), learning from everything before each and testing on the year after, then combine. More evidence, and it fixes the "one test period" limitation.
- For a fair retrial, the models are frozen and the pass marks written before the run.

## 2026-11-16 · Phase 5b: multi-season scoreboard

- **Seasons:** cutoffs 1 Jan 2009, 2010, 2011, 2012, each tested on the 12 months after it. Data ends Oct 2013, so 2013 cannot be a full season. 2009 was added after a count showed it has 467 test investors (pass mark: at least 300). Denzel chose four seasons over three.
- **Tuning per season (Denzel's choice):** the old practice year (2011) is the 2011 season's test year, so reusing those settings would let the 2011 season see its own answers. Each season reruns the same recipe (same grids, same rules) on the year before its cutoff. Practice cutoff is now a parameter of tuning; the main scoreboard still uses 2011.
- **Combining:** paired per-investor differences from every season, with a cluster bootstrap that resamples whole investors. Checked by tests: with one row per investor it equals the plain bootstrap, and copying every row does not tighten the interval (the plain bootstrap gets fooled by that).
- **Honesty note:** the 2012 season overlaps the test already seen while choosing models, so the verdict is also reported on 2009 to 2011 alone.
- **Pass marks, written before the run:** 1 no leaks, 2 at least 300 test investors per season, 3 same seed same numbers, 4a Thompson on the hybrid beats label similarity over all seasons, 4b same on the fresh seasons, 5 whole CI run under 12 minutes.
- **Result:** 1, 2 and 3 met. 4a failed (+0.0012, CI -0.0029 to +0.0053) and 4b failed (+0.0023, CI -0.0028 to +0.0073). Recorded as strict expected-to-fail tests. **Not promoted:** label similarity stays the starter.
- The Phase 5 edge (+0.0044 on one test period) mostly shrank when replayed over four seasons. That is the reason for building a series: one good game can be luck.
- Run note: the full real-data suite is about 5 minutes locally (seasons about 2.5 of that).

## 2026-11-16 · Phase 6: explain each pick

- **Scope:** only the starter (label similarity) is explained. The bench models never reach a user, and their scores come from hidden profiles that cannot be split honestly; any "why" for them would be a guess. A promoted model gets its own explainer then.
- **Method:** a startup has at most two ticks, so cosine similarity splits exactly: part(label) = profile[label] / (|profile| x |startup|). The explanation is the score taken apart, not a story written afterwards (faithful, not just plausible).
- Evidence shown: how many past deals share each label, with up to two examples by name. Junk names (blank or Excel's "#NAME?") fall back to the web address.
- A pick sharing no label would be labelled "shown by chance", never given a made-up reason.
- **Pass marks, written before the run:** 1 parts add up to the score, 2 no invented evidence, 3 nothing from the test period cited, 4 every top-10 pick explained, 5 biggest reason first. **All met** on 8,280 picks (biggest add-up error 1.1e-16). Zero picks were shown by chance.
- **Not checked:** whether a real investor finds the explanations convincing. That needs people, out of scope for now.
- **Finding:** startups missing a label are 2% of the board but 36% of the picks. A one-tick checklist is shorter (length 1 instead of about 1.41), so one match is not diluted. Possible fixes for later (each needs its own pass marks and a scoreboard run): give a missing label its own "unknown" tick, or always divide by the full two-label length.
- Test setup slip, fixed before any code changed: a hand-made test investor had no software deals, so the test could not see the line it checked.

## 2026-11-19 · Phase 7: the quality gate

- **Rules, chosen by Denzel before any challenger was judged:** 1 proven win over all seasons (and not a real loss on the fresh seasons); 2 fairness relative to the champion, at most 5 points below its new-startup share in every season; 3 explainable, with an exact explainer.
- **Why relative fairness (option A):** an absolute rule ("at least the board share") would fail the current champion itself (52 to 63% against board shares of 57 to 64%). A gate that rejects the current starter is broken. Weakness: it only stops things getting worse, it does not force them to get better.
- **Why explainable is required:** the product promise is a pick plus an honest why. Cost: models like matrix factorisation cannot become champion until someone builds an exact explainer for them.
- **Mechanics:** `backend/champion.json` holds the champion and a version. A promotion bumps the version by exactly one. The `quality-gate` job always runs (a skipped required check would block merging forever) and finishes in seconds when nothing changed. The release workflow on `main` judges a change again before tagging `model-vN`.
- The current starter becomes model v1 when this phase reaches `main` (no contest: it earned the spot in Phases 2 to 6).
- The multi-season run is cached within one test run, so the gate tests and the season tests share one scoring (about 2.5 minutes, not 5). The season repeatability test now recomputes from scratch on purpose.
- **Pass marks, written before the run:** 1 rejects every recorded loser with the right rule, 2 can say yes to a planted winner (a cheat that sees the answers), 3 no change passes without running the scoreboard, 4 an unknown model fails loudly, 5 a release is tagged only when the champion changes. **All met.**

## 2026-11-19 · Phase 7b: the missing-label fix, first challenger through the gate

- **Fix:** divide every startup by the full checklist length (square root of 2 for two label columns), so a missing label counts as a non-match. Rejected alternative: an "unknown" label, which would let investors "match" on missing data and reward startups for being unlabelled.
- Kept as a new model, `label_similarity_full`, next to the champion, so the gate can compare them. Its explainer is the Phase 6 explainer: the model tells it which length to divide by, so the two can never disagree. It also says "No state on record, which counts as no match" when a label is missing.
- **Pass marks, written before the run:** 1 identical scores for fully labelled startups (exact) and old divided by root 2 for one missing label; 2 missing-label picks under 10% (from 36%); 3 the explainer passes all five Phase 6 checks; 4 nothing else breaks. **All met** (missing-label picks: 0%).
- **Gate verdict: PASSED.** All seasons +0.0008 (CI +0.0001 to +0.0015); fresh seasons +0.0006 (CI -0.0000 to +0.0012); new-startup share at most 4 points below the champion (2011: 59% vs 63%); explainable.
- **Caveats, kept visible:** the win is small (about 3% of NDCG@10). It is provable at this size because the paired differences are low-noise: the two models only disagree where a label is missing. Two rules pass with thin margins. It overshoots: missing-label startups go from over-picked (36%) to never picked (0%, against 2% of the board). The fix was designed after seeing 2012 explanations; the fresh seasons show it is not a loss there, but its proven win leans on all four seasons.
- **Decision:** the rules were set before the contest, and the challenger met them, so it is promoted to champion v2. Changing the rules now because the margin is thin would be moving the hoop after the shot.

## 2026-11-19 · Phase 8a: the API

- **FastAPI + Uvicorn**, local only. Licences checked: FastAPI, Pydantic MIT; Starlette, Uvicorn, httpx BSD-3; certifi (a dependency) MPL-2.0, fine as an unchanged dependency. All free.
- **One engine, not a copy:** the API uses the same data, draft board, ranking and tie-break order as the scoreboard. Pass mark 1 checks all 828 test investors pick for pick.
- **Champion from `champion.json`:** the API serves whatever the gate last crowned, and refuses a champion without an exact explainer (the API never shows a pick without an honest why).
- **Investor ids go in a query parameter**, not the path, because they contain slashes (`/financial-organization/sequoia-capital`).
- **Hindsight check:** for the 2012 test investors, each pick says whether they really backed it later. Evaluation made visible for a demo.
- **Errors:** unknown investor 404 with what to do next, bad input 422, unexpected failures logged and answered with a short 500 (no stack trace). Pass mark 4 checks that bad input never causes a 500.
- **Finding:** big investors' top picks often tie (same score, same reason), because two labels cannot separate startups in the same category and state. The order among ties is the random tie-break. Recorded as the label-only ceiling; Version 2 (descriptions) is the fix.
- Known warning: Starlette now prefers a newer test client than httpx. Harmless today; revisit if it breaks.
- Slip fixed: the Phase 7b README edit had dropped the line "Pass marks are written down before each run..."; restored here.
- Pass marks 1 to 5 all met. 174 passed, 1 skipped, 7 expected failures locally.

## 2026-11-19 · Phase 8b: the demo page (Scout Desk)

- **Design rules from Denzel** (the thinking behind a Funema team page, used as principles only, none of its colours, fonts or wording): hide the maths and show the meaning; every claim shows its why; uncertainty is a real state; an app, not a long page; one accent colour for one job; words from the user's side; a quality floor (4.5:1 contrast, 44px targets, keyboard, phone).
- **Challenges raised before building:** our data has no stage and no Africa (US startups, sector and state, up to 2012), so evidence can only be "same sector", "same state" and "like these past investments". "Strong fit" was replaced by plain words, because the model measures resemblance to past investments, not how likely an investor is to invest.
- **Prototype first:** Denzel asked to see it before any branch existed. Built from real API answers outside the repo, screenshotted and critiqued in rounds. When Denzel said he did not understand it, the words were simplified (one sentence per screen, "investments" not "deals", "They invested" not "Backed later", detail moved behind "How we worked this out") and he approved that version.
- **Palette:** paper #F3F5F2, panel #FFFFFF, ink #1B201B, quiet #566056, line #D7DDD5, one accent (field olive #465724) used only for the main button, the key result line and keyboard focus. A test checks the accent is used nowhere else. **Type:** Barlow Semi Condensed 600 for headings and states, Public Sans for text, both OFL-1.1 and bundled locally (no font server).
- **Stack:** Vite + plain TypeScript, Vitest + jsdom (all MIT), TypeScript 5 (Apache-2.0; pinned instead of the brand-new 7). In development Vite passes /api calls to the Python API, so no cross-origin requests.
- **Pass marks, written before building:** 1 strict TypeScript; 2 faithful to the API; 3 every failure handled with a next step; 4 works end to end on the real API; 5 usable (keyboard, 44px, contrast, phone); 6 a frontend CI job. **All met.**
- **Bugs caught by the checks:** "Try again" after a failed search went back to Overview instead of retrying (fixed: the page remembers the last action); the search box shrank below 44px on the start screen when the list was long (fixed, and now measured on the start screen too).
- The end-to-end browser run happens in MARA's workspace (the browser in the Claude app cannot reach servers in the Mac's shell); it is not in CI yet.
