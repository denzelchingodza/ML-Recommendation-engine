// Pass mark 2 (faithful) and the design rules, on real API answers recorded from the running API.
import { describe, expect, it } from "vitest";
import picksAustinRaw from "./fixtures/picks-austin.json";
import picksSequoiaRaw from "./fixtures/picks-sequoia.json";
import profileAustinRaw from "./fixtures/profile-austin.json";
import type { Pick, Picks, Profile } from "./types";
import { confident, matchWords, overview, problem, results, review } from "./view";

const austin = picksAustinRaw.body as Picks;
const sequoia = picksSequoiaRaw.body as Picks;
const profile = profileAustinRaw.body as Profile;
const text = (html: string) => { const d = document.createElement("div"); d.innerHTML = html; return d.textContent ?? ""; };

describe("results are faithful to the API", () => {
  it("shows every confident pick by name, in the API's order", () => {
    const html = results(austin, profile, { hindsight: false, how: false });
    const d = document.createElement("div"); d.innerHTML = html;
    const shown = [...d.querySelectorAll(".results .name")].map((n) => n.textContent);
    expect(shown).toEqual(confident(austin.picks).map((p) => p.name));
  });

  it("marks exactly the startups the investor really backed, and only when asked", () => {
    const off = results(austin, profile, { hindsight: false, how: false });
    const on = results(austin, profile, { hindsight: true, how: false });
    const backed = austin.picks.filter((p) => p.backed_later).length;
    expect(text(off)).not.toContain("They invested");
    expect(text(on).split("They invested").length - 1).toBe(backed);
    expect(text(on)).toContain(`${backed} of 10suggestions Austin Ventures really invested in`);
  });

  it("says plainly when none of the picks were backed later", () => {
    expect(text(results(sequoia, null, { hindsight: true, how: false }))).toContain("0 of 10suggestions Sequoia Capital really invested in");
  });

  it("leads with one plain sentence", () => {
    expect(text(results(austin, profile, { hindsight: false, how: false })))
      .toContain("These 10 startups look like Austin Ventures' past investments: software companies in Texas.");
  });
});

describe("hide the maths, show the meaning", () => {
  it("the model speaks for itself: no screen says 'we' or 'our'", () => {
    const screens = [results(austin, profile, { hindsight: true, how: true }), results(sequoia, null, { hindsight: true, how: true }),
      overview(profile), review(austin, { hindsight: true, how: true })];
    for (const html of screens) expect(text(html)).not.toMatch(/\b(we|our|us)\b/i);
  });

  it("the Searched step shows how many startups the model really searched", () => {
    expect(text(results(austin, profile, { hindsight: false, how: false }))).toContain(`${austin.searched.toLocaleString("en")} startups`);
  });

  it("shows no scores or percentages unless 'Show the maths' is open", () => {
    for (const html of [results(austin, profile, { hindsight: true, how: false }), overview(profile), review(austin, { hindsight: true, how: false })]) {
      expect(text(html)).not.toMatch(/\d\.\d\d|%/);
    }
  });

  it("shows the model's working as steps, with the raw score behind 'Show the maths'", () => {
    const t = text(results(austin, profile, { hindsight: false, how: true }));
    expect(t).toContain("How the model worked this out");
    expect(t).toContain("27 equally strong");
    expect(t).toContain("It can't tell those 27 apart, so it shows a random 10");
    expect(t).toContain("Score 0.84 out of 1 = Texas 0.63 + Software 0.20");
    expect(t).toContain("Adometry, Alereon, ACTIVE Network and AlterPoint");
  });

  it("describes the investor in everyday words", () => {
    const t = text(overview(profile));
    expect(t).toContain("Austin Ventures usually invests in software companies in Texas.");
    expect(t).toContain("59 of its 85 investments before 2012 were in Texas, and 19 were software companies.");
    expect(t).toContain("Find startups for this investor");
  });
});

describe("states and evidence", () => {
  const base = austin.picks[0]!;
  const state = base.reasons.find((r) => r.label === "state")!;
  it.each([
    [{ ...base, match: "close" } as Pick, "Same sector and state"],
    [{ ...base, match: "partial", reasons: [state] } as Pick, "Same state only"],
    [{ ...base, match: "possible" } as Pick, "Possible"],
  ])("%#: plain words for each match state", (pick, words) => {
    expect(matchWords(pick)).toBe(words);
  });

  it("puts unsure picks in Review, never mixed into Results", () => {
    const mixed: Picks = { ...austin, picks: [austin.picks[0]!, { ...austin.picks[1]!, match: "possible", name: "Unsure Co" }] };
    expect(text(results(mixed, profile, { hindsight: false, how: false }))).not.toContain("Unsure Co");
    expect(text(review(mixed, { hindsight: false, how: false }))).toContain("Unsure Co");
  });

  it("an empty Review explains itself instead of showing an empty list", () => {
    expect(text(review(austin, { hindsight: false, how: false }))).toContain("Nothing to double-check");
  });
});

describe("problems are screens with a next step", () => {
  it.each([
    [{ kind: "unreachable" } as const, "Couldn't load startups right now", "Try again"],
    [{ kind: "server" } as const, "The recommender ran into a problem", "Try again"],
    [{ kind: "unknown_investor" } as const, "That investor isn't in the data", "Choose another investor"],
    [{ kind: "not_enough_history", pastDeals: 3, needed: 5 } as const, "Not enough history on ACT Venture Capital yet", "Choose another investor"],
    [{ kind: "bad_request", message: "" } as const, "The page asked for something the recommender can't answer", "Choose another investor"],
  ])("%#: %s", (p, title, action) => {
    const t = text(problem(p, "ACT Venture Capital"));
    expect(t).toContain(title);
    expect(t).toContain(action);
    expect(t).not.toMatch(/sorry|oops/i);
  });

  it("escapes anything that looks like code in names", () => {
    const nasty: Picks = { ...austin, name: "<img src=x onerror=alert(1)>", picks: [{ ...austin.picks[0]!, name: "<b>bold</b>" }] };
    const html = results(nasty, null, { hindsight: false, how: false });
    expect(html).not.toContain("<img"); expect(html).not.toContain("<b>bold");
  });
});
