// The Scout Desk app: investor list on the left, Overview / Results / Review tabs on the right.
// One main action per screen. Every failure becomes a named screen with a next step.

import { ApiError, type Api, type Problem } from "./api";
import type { Champion, InvestorSummary, Picks, Profile } from "./types";
import { confident, investorList, modelLine, overview, problem, results, review, unsure } from "./view";
import { escapeHtml as e, plural } from "./words";

type Tab = "overview" | "results" | "review";

interface State {
  champion: Champion | null;
  query: string;
  investors: InvestorSummary[] | null;
  listProblem: boolean;
  currentId: string | null;
  currentName: string | null;
  profile: Profile | null;
  picks: Picks | null;
  tab: Tab;
  hindsight: boolean;
  how: boolean;
  problem: Problem | null;
  loading: string | null;
  lastAction: "profile" | "picks" | null; // what 'Try again' repeats
}

const SHELL = `
<header class="top"><h1>Scout Desk</h1>
  <span class="tag">Pick an investor and the model suggests new startups that look like the ones they've backed before.</span>
  <span class="model" id="model"></span></header>
<div class="app" id="app">
  <aside class="side" aria-label="Investors">
    <label for="q">Find an investor</label>
    <input id="q" type="search" placeholder="Name, e.g. Sequoia" autocomplete="off">
    <ul id="list" aria-live="polite"></ul>
  </aside>
  <section class="main">
    <div class="who"><button type="button" class="back" id="back">All investors</button><h2 id="iname"></h2><p id="imeta"></p></div>
    <div class="tabs" role="tablist" aria-label="Investor views" id="tabs">
      <button type="button" role="tab" id="t-overview" data-tab="overview" aria-controls="panel">Overview</button>
      <button type="button" role="tab" id="t-results" data-tab="results" aria-controls="panel">Results</button>
      <button type="button" role="tab" id="t-review" data-tab="review" aria-controls="panel">Review</button>
    </div>
    <div class="panel" id="panel" role="tabpanel" tabindex="0" aria-live="polite"></div>
  </section>
</div>`;

const WELCOME = `<div class="empty"><h3>Choose an investor to start</h3>
  <p>Pick one from the list, or search by name. The model shows what they usually invest in, then suggests new startups that fit.</p></div>`;

export function mountApp(root: HTMLElement, api: Api): { ready: Promise<void> } {
  root.innerHTML = SHELL;
  const $ = <T extends HTMLElement>(id: string) => root.querySelector<T>(`#${id}`)!;
  const s: State = { champion: null, query: "", investors: null, listProblem: false, currentId: null, currentName: null,
    profile: null, picks: null, tab: "overview", hindsight: false, how: false, problem: null, loading: null, lastAction: null };

  const asProblem = (err: unknown): Problem => (err instanceof ApiError ? err.problem : { kind: "server" });

  function renderList(): void {
    const list = $("list");
    if (s.listProblem) {
      list.innerHTML = `<li class="none">Couldn't load investors. Check the recommender is running.
        <button type="button" class="secondary small" id="list-retry">Try again</button></li>`;
    } else if (s.investors === null) {
      list.innerHTML = `<li class="none">Loading investors</li>`;
    } else {
      list.innerHTML = investorList(s.investors, s.currentId, s.query);
    }
  }

  function renderMain(): void {
    $("model").textContent = modelLine(s.champion);
    root.querySelector("#app")!.classList.toggle("picked", s.currentId !== null);
    $("iname").textContent = s.currentName ?? "";
    $("imeta").textContent = s.profile ? `${plural(s.profile.past_deals, "investment", "investments")} before ${s.profile.cutoff.slice(0, 4)}` : "";
    const found = s.picks !== null;
    const counts: Record<Tab, string> = {
      overview: "Overview",
      results: found ? `Results (${confident(s.picks!.picks).length})` : "Results",
      review: found ? `Review (${unsure(s.picks!.picks).length})` : "Review",
    };
    root.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => {
      const t = b.dataset.tab as Tab;
      b.textContent = counts[t];
      b.setAttribute("aria-selected", String(t === s.tab));
      b.tabIndex = t === s.tab ? 0 : -1;
      b.disabled = s.currentId === null || (t !== "overview" && !found);
    });
    const panel = $("panel");
    if (s.currentId === null) panel.innerHTML = WELCOME;
    else if (s.loading) panel.innerHTML = `<p class="note">${e(s.loading)}</p>`;
    else if (s.problem) panel.innerHTML = problem(s.problem, s.currentName);
    else if (s.tab === "overview" && s.profile) panel.innerHTML = overview(s.profile);
    else if (s.tab === "results" && s.picks) panel.innerHTML = results(s.picks, s.profile, s);
    else if (s.tab === "review" && s.picks) panel.innerHTML = review(s.picks, s);
  }

  function render(focusId?: string): void {
    renderList();
    renderMain();
    if (focusId) root.querySelector<HTMLElement>(`#${focusId}`)?.focus();
  }

  async function loadList(): Promise<void> {
    const query = s.query;
    try {
      const found = await api.search(query, 30);
      if (query !== s.query) return; // a newer search is on its way
      s.investors = found; s.listProblem = false;
    } catch {
      s.listProblem = true;
    }
    renderList();
  }

  async function choose(id: string): Promise<void> {
    const summary = s.investors?.find((i) => i.id === id);
    Object.assign(s, { currentId: id, currentName: summary?.name ?? id, profile: null, picks: null, tab: "overview",
      hindsight: false, how: false, problem: null, loading: `Loading ${summary?.name ?? "investor"}`, lastAction: "profile" });
    render();
    try {
      s.profile = await api.profile(id);
      s.currentName = s.profile.name;
      if (!s.profile.enough_history) {
        s.problem = { kind: "not_enough_history", pastDeals: s.profile.past_deals, needed: s.profile.needed };
      }
    } catch (err) {
      s.problem = asProblem(err);
    }
    s.loading = null;
    render(s.problem ? (s.problem.kind === "unreachable" || s.problem.kind === "server" ? "retry" : "other") : "find");
  }

  async function find(): Promise<void> {
    if (!s.currentId) return;
    s.loading = `Finding startups for ${s.currentName ?? "this investor"}`; s.problem = null; s.lastAction = "picks";
    render();
    try {
      s.picks = await api.picks(s.currentId, 10);
      s.tab = "results";
    } catch (err) {
      s.problem = asProblem(err);
    }
    s.loading = null;
    render(s.problem ? "retry" : "t-results");
  }

  function setTab(tab: Tab, focus = true): void {
    s.tab = tab;
    render(focus ? `t-${tab}` : undefined);
  }

  // Clicks, delegated so they keep working after each render
  root.addEventListener("click", (ev) => {
    const target = (ev.target as HTMLElement).closest<HTMLElement>("button");
    if (!target || target.hasAttribute("disabled")) return;
    if (target.dataset.id) { void choose(target.dataset.id); return; }
    if (target.dataset.tab) { setTab(target.dataset.tab as Tab); return; }
    switch (target.id) {
      case "find": void find(); break;
      case "hindsight": s.hindsight = !s.hindsight; render("hindsight"); break;
      case "how": s.how = !s.how; render("how"); break;
      case "to-review": setTab("review"); break;
      case "to-results": setTab("results"); break;
      case "retry":
        if (s.lastAction === "picks") void find();
        else if (s.currentId) void choose(s.currentId);
        break;
      case "list-retry": s.investors = null; s.listProblem = false; renderList(); void loadList(); break;
      case "other": case "back":
        s.currentId = null; s.currentName = null; s.profile = null; s.picks = null; s.problem = null;
        render("q"); break;
    }
  });

  // Arrow keys move between tabs (the usual tablist pattern)
  $("tabs").addEventListener("keydown", (ev) => {
    if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
    const tabs = [...root.querySelectorAll<HTMLButtonElement>("#tabs button:not(:disabled)")];
    const i = tabs.indexOf(document.activeElement as HTMLButtonElement);
    if (i < 0) return;
    const next = tabs[(i + (ev.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]!;
    setTab(next.dataset.tab as Tab);
    ev.preventDefault();
  });

  let timer: ReturnType<typeof setTimeout> | undefined;
  $<HTMLInputElement>("q").addEventListener("input", (ev) => {
    s.query = (ev.target as HTMLInputElement).value;
    clearTimeout(timer);
    timer = setTimeout(() => void loadList(), 200);
  });

  render();
  const ready = Promise.all([
    api.champion().then((c) => { s.champion = c; }).catch(() => { /* the header just stays quiet */ }),
    loadList(),
  ]).then(() => render());
  return { ready };
}
