// Pass marks 3 and 5: the whole flow by mouse and keyboard, and every failure handled, on real recorded answers.
import { beforeEach, describe, expect, it } from "vitest";
import { ApiError, type Api } from "./api";
import { mountApp } from "./app";
import champion from "./fixtures/champion.json";
import picksAustin from "./fixtures/picks-austin.json";
import profileAct from "./fixtures/profile-act.json";
import profileAustin from "./fixtures/profile-austin.json";
import type { Champion, InvestorSummary, Picks, Profile } from "./types";

const AUSTIN = "/financial-organization/austin-ventures", ACT = "/financial-organization/act-venture-capital";
const list: InvestorSummary[] = [
  { id: AUSTIN, name: "Austin Ventures", past_deals: 85, enough_history: true },
  { id: ACT, name: "ACT Venture Capital", past_deals: 3, enough_history: false },
];
const tick = () => new Promise((r) => setTimeout(r, 0));

function fakeApi(overrides: Partial<Api> = {}): Api {
  return {
    champion: async () => champion.body as Champion,
    search: async (q) => list.filter((i) => i.name.toLowerCase().includes(q.toLowerCase())),
    profile: async (id) => (id === ACT ? profileAct.body : profileAustin.body) as Profile,
    picks: async () => picksAustin.body as Picks,
    ...overrides,
  };
}

let root: HTMLElement;
const $ = (sel: string) => root.querySelector<HTMLElement>(sel)!;
const click = async (el: HTMLElement) => { el.click(); await tick(); await tick(); };

beforeEach(() => { document.body.innerHTML = '<div id="root"></div>'; root = document.getElementById("root")!; });

describe("the main flow", () => {
  it("goes from investor to results to 'did it work?'", async () => {
    await mountApp(root, fakeApi()).ready;
    expect($("#model").textContent).toBe("Model v2, checked by the quality gate");
    expect($("#panel").textContent).toContain("Choose an investor to start");

    await click($(`[data-id="${AUSTIN}"]`));
    expect($("#panel").textContent).toContain("Austin Ventures usually invests in software companies in Texas.");
    expect(($("#t-results") as HTMLButtonElement).disabled).toBe(true);   // nothing to show before the main action
    expect(document.activeElement?.id).toBe("find");                       // focus goes to the one main action

    await click($("#find"));
    expect($("#t-results").getAttribute("aria-selected")).toBe("true");
    expect($("#t-results").textContent).toBe("Results (10)");
    expect($("#t-review").textContent).toBe("Review (0)");
    expect(root.querySelectorAll(".results li")).toHaveLength(10);

    await click($("#hindsight"));
    expect($("#panel").textContent).toContain("3 of 10suggestions Austin Ventures really invested in");
    expect(document.activeElement?.id).toBe("hindsight");
  });

  it("moves between tabs with the arrow keys", async () => {
    await mountApp(root, fakeApi()).ready;
    await click($(`[data-id="${AUSTIN}"]`));
    await click($("#find"));
    $("#t-results").focus();
    $("#tabs").dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    expect(document.activeElement?.id).toBe("t-review");
    expect($("#panel").textContent).toContain("Nothing to double-check");
  });

  it("searches by name and says so when nothing matches", async () => {
    await mountApp(root, fakeApi()).ready;
    const q = $("#q") as HTMLInputElement;
    q.value = "zzz"; q.dispatchEvent(new Event("input"));
    await new Promise((r) => setTimeout(r, 260));
    expect($("#list").textContent).toContain('No investor matches "zzz"');
  });
});

describe("uncertainty and failures are real screens", () => {
  it("not enough history: a proper screen with a next step, and no tabs to click", async () => {
    await mountApp(root, fakeApi()).ready;
    await click($(`[data-id="${ACT}"]`));
    expect($("#panel").textContent).toContain("Not enough history on ACT Venture Capital yet");
    expect($("#panel").textContent).toContain("ACT Venture Capital has 3.");
    expect(($("#t-results") as HTMLButtonElement).disabled).toBe(true);
    await click($("#other"));
    expect(document.activeElement?.id).toBe("q");
  });

  it("the recommender not answering: says so, and 'Try again' really tries again", async () => {
    let calls = 0;
    const api = fakeApi({ picks: async () => {
      calls += 1;
      if (calls === 1) throw new ApiError({ kind: "unreachable" });
      return picksAustin.body as Picks;
    } });
    await mountApp(root, api).ready;
    await click($(`[data-id="${AUSTIN}"]`));
    await click($("#find"));
    expect($("#panel").textContent).toContain("Couldn't load startups right now");
    await click($("#retry"));
    expect(root.querySelectorAll(".results li")).toHaveLength(10);
  });

  it("the investor list failing to load: says so, with a retry, never an empty list", async () => {
    let calls = 0;
    const api = fakeApi({ search: async () => { calls += 1; if (calls === 1) throw new ApiError({ kind: "unreachable" }); return list; } });
    await mountApp(root, api).ready;
    expect($("#list").textContent).toContain("Couldn't load investors");
    await click($("#list-retry"));
    expect($("#list").textContent).toContain("Austin Ventures");
  });

  it("every button and tab is a real button (keyboard reachable)", async () => {
    await mountApp(root, fakeApi()).ready;
    await click($(`[data-id="${AUSTIN}"]`));
    await click($("#find"));
    for (const el of root.querySelectorAll("[id='find'], [id='hindsight'], [id='how'], [role='tab'], [data-id]")) {
      expect(el.tagName).toBe("BUTTON");
    }
  });
});
