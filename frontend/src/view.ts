// Turns API answers into screens. Pure functions returning HTML, so every screen can be tested.
// Design rules (agreed 19 Nov 2026): plain states instead of scores; every claim shows its evidence;
// uncertainty is its own screen; the model shows its working as steps, raw scores only behind "Show the maths".
// The model speaks for itself (never "we"). All API text is escaped.

import type { Problem } from "./api";
import type { Champion, InvestorSummary, Pick, Picks, Profile } from "./types";
import { capitalise, escapeHtml as e, listOf, plural, possessive } from "./words";

export const year = (cutoff: string): string => cutoff.slice(0, 4);

export function modelLine(champion: Champion | null): string {
  return champion ? `Model v${champion.version}, checked by the quality gate` : "";
}

export function investorList(items: InvestorSummary[], currentId: string | null, query: string): string {
  if (!items.length) {
    return `<li class="none">No investor matches "${e(query)}". Try part of the name, like "Capital".</li>`;
  }
  return items.map((i) => `<li><button type="button" data-id="${e(i.id)}"${i.id === currentId ? ' aria-current="true"' : ""}>
    <span>${e(i.name)}</span><small>${plural(i.past_deals, "investment", "investments")}</small></button></li>`).join("");
}

export function overview(p: Profile): string {
  const y = year(p.cutoff);
  const st = p.top_state, sec = p.top_sector;
  const habit = st && sec ? `${e(p.name)} usually invests in ${e(sec.name)} companies in ${e(st.name)}.`
    : st ? `${e(p.name)} usually invests in ${e(st.name)}.`
    : sec ? `${e(p.name)} usually invests in ${e(sec.name)} companies.`
    : `${e(p.name)} has no sector or state on record for its past investments.`;
  const facts = [st ? `${st.count} of its ${p.past_deals} investments before ${y} were in ${e(st.name)}` : "",
                 sec ? `${sec.count} were ${e(sec.name)} companies` : ""].filter(Boolean).join(", and ");
  return `<p class="report">${habit}</p>
  <p class="note">${facts ? `${facts}. ` : ""}The model will look for new startups like that.</p>
  <button type="button" class="primary" id="find">Find startups for this investor</button>`;
}

export function matchWords(p: Pick): string {
  if (p.match === "close") return "Same sector and state";
  if (p.match === "partial") return p.reasons[0]?.label === "state" ? "Same state only" : "Same sector only";
  return "Possible";
}

const TICK = '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2"/></svg>';

const reasonOf = (p: Pick, label: "state" | "category") => p.reasons.find((r) => r.label === label);

export function confident(picks: Pick[]): Pick[] { return picks.filter((p) => p.match !== "possible"); }
export function unsure(picks: Pick[]): Pick[] { return picks.filter((p) => p.match === "possible"); }

function keyLine(data: Picks, list: Pick[]): string {
  const n = list.length, who = possessive(data.name);
  const first = list[0]!;
  const sameKind = list.every((p) => p.match === "close" && p.reasons.map((r) => r.value).join() === first.reasons.map((r) => r.value).join());
  const sec = reasonOf(first, "category"), st = reasonOf(first, "state");
  const what = sameKind && sec && st ? `: ${e(sec.value_name)} companies in ${e(st.value_name)}` : "";
  const subject = n === 1 ? "This startup looks" : `These ${n} startups look`;
  return `${subject} like ${e(who)} past investments${what}.`;
}

function howPanel(data: Picks, list: Pick[], profile: Profile | null, maths: boolean): string {
  const y = year(data.cutoff), n = list.length, first = list[0]!;
  const deals = profile ? profile.past_deals : data.past_deals;
  const st = profile?.top_state ?? null, sec = profile?.top_sector ?? null;
  const groups = new Map<number, Pick[]>();
  for (const p of list) groups.set(p.score, [...(groups.get(p.score) ?? []), p]);
  const oneGroup = groups.size === 1;
  const shared = listOf(first.reasons.map((r) => r.value_name));
  const examples = listOf([...new Set(list.flatMap((p) => p.reasons.flatMap((r) => r.example_names)))].slice(0, 4));
  const pattern = st && sec ? `${e(st.name)} and ${e(sec.name)}` : st ? e(st.name) : sec ? e(capitalise(sec.name)) : "No clear pattern";
  const patternLine = [st ? `${st.count} of ${deals} in ${e(st.name)}` : "", sec ? `${sec.count} ${e(sec.name)} companies` : ""].filter(Boolean).join(", ");
  const steps: [string, string, string][] = [
    ["Read the history", plural(deals, "investment", "investments"), `Everything ${e(data.name)} invested in before ${y}`],
    ["Found the pattern", pattern, patternLine || "Too few labels on record to see a pattern"],
    ["Searched", `${data.searched.toLocaleString("en")} startups`, `Every startup that raised money from ${y} on, except ones it already backed`],
    oneGroup && first.equally_strong > n
      ? ["Matched", `${first.equally_strong.toLocaleString("en")} equally strong`, `All share ${e(shared)} with its past investments`]
      : ["Matched", `${n} closest`, "Ranked by how much each shares with its past investments"],
    oneGroup && first.equally_strong > n
      ? ["Picked", `${n} at random`, `It can't tell those ${first.equally_strong.toLocaleString("en")} apart, so it shows a random ${n}`]
      : ["Picked", `${n} shown`, "The strongest matches, best first"],
  ];
  const mathLines = [...groups.values()].map((g) => {
    const p = g[0]!;
    const parts = p.reasons.map((r) => `${e(capitalise(r.value_name))} ${r.part.toFixed(2)}`).join(" + ");
    return `<p>Score ${p.score.toFixed(2)} out of 1${parts ? ` = ${parts}` : ""}${groups.size > 1 ? ` (${g.length} startups)` : ""}</p>`;
  }).join("");
  return `<section class="work" aria-labelledby="work-title"><h3 id="work-title">How the model worked this out</h3>
    <ol class="steps">${steps.map(([label, figure, line], i) => `<li><span class="num">${i + 1}</span><span class="label">${label}</span>
      <strong class="figure">${figure}</strong><span class="line">${line}</span></li>`).join("")}</ol>
    ${examples ? `<p class="looks">Looks like past investments such as ${e(examples)}.</p>` : ""}
    <button type="button" class="link" id="how" aria-expanded="${maths}" aria-controls="maths-panel">${maths ? "Hide the maths" : "Show the maths"}</button>
    ${maths ? `<div class="how" id="maths-panel">${mathLines}<p>Each label's part is how often it appears in the investor's past, scaled so a perfect match scores 1.</p></div>` : ""}
  </section>`;
}

export interface ResultsOptions { hindsight: boolean; how: boolean }

const facts = (p: Pick): string => {
  const sec = reasonOf(p, "category"), st = reasonOf(p, "state");
  if (!sec && !st) return `<span class="facts">Nothing in common on record</span>`;
  return `<span class="facts">${sec ? `<span>Sector <b>${e(capitalise(sec.value_name))}</b></span>` : ""}${st ? `<span>State <b>${e(st.value_name)}</b></span>` : ""}</span>`;
};

function rowsOf(list: Pick[], hindsight: boolean): string {
  return `<ol class="results">${list.map((p) => `<li><span class="name">${e(p.name)}</span><span class="state">${matchWords(p)}</span>
    ${facts(p)}${hindsight && p.backed_later ? `<span class="later">${TICK}They invested</span>` : "<span></span>"}</li>`).join("")}</ol>`;
}

export function results(data: Picks, profile: Profile | null, opts: ResultsOptions): string {
  const list = confident(data.picks);
  if (!list.length) {
    return `<div class="empty"><h3>No close matches for ${e(data.name)}</h3>
      <p>Startups the model is less sure about are in the Review tab.</p>
      <button type="button" class="secondary" id="to-review">Go to Review</button></div>`;
  }
  const y = year(data.cutoff), n = list.length, hits = list.filter((p) => p.backed_later).length;
  const later = profile?.later_investments;
  const did = data.hindsight_available ? `<section class="did" aria-labelledby="did-title"><h3 id="did-title">Did it work?</h3>
    <p>The model only used investments up to ${y}. See what ${e(data.name)} really did from ${y} on.</p>
    <button type="button" class="secondary" id="hindsight" aria-pressed="${opts.hindsight}">${opts.hindsight ? "Hide what really happened" : "Check against what really happened"}</button>
    ${opts.hindsight ? `<div class="stats" role="status"><div><strong>${hits} of ${n}</strong><span>suggestions ${e(data.name)} really invested in</span></div>
      ${later ? `<div><strong>${later - hits}</strong><span>other startups ${e(data.name)} invested in instead</span></div>` : ""}</div>` : ""}</section>` : "";
  return `<div class="results-layout"><div class="main-col"><p class="key">${keyLine(data, list)}</p>${rowsOf(list, opts.hindsight)}</div>
    <div class="side-col">${howPanel(data, list, profile, opts.how)}${did}</div></div>`;
}

export function review(data: Picks, opts: ResultsOptions): string {
  const list = unsure(data.picks);
  if (!list.length) {
    return `<div class="empty"><h3>Nothing to double-check</h3>
      <p>All of these startups match on sector, state or both. Startups the model is less sure about, for example ones missing their sector or state, would show up here.</p>
      <button type="button" class="secondary" id="to-results">Back to results</button></div>`;
  }
  return `<p class="key">${plural(list.length, "startup needs", "startups need")} a second look</p>
    <p class="note">The model is less sure about these: they're missing a sector or state on record, or share little with ${e(possessive(data.name))} past investments.</p>
    ${rowsOf(list, opts.hindsight)}`;
}

export function problem(p: Problem, investorName: string | null): string {
  const who = investorName ? e(investorName) : "this investor";
  const screens: Record<Problem["kind"], [string, string, string, string]> = {
    unreachable: ["Couldn't load startups right now", "The recommender didn't answer. Check it is running, then try again.", "retry", "Try again"],
    server: ["The recommender ran into a problem", "It has been logged on the recommender's side. Try again in a moment.", "retry", "Try again"],
    unknown_investor: ["That investor isn't in the data", "They may have no investments before 2012 on record. Choose another investor.", "other", "Choose another investor"],
    not_enough_history: [`Not enough history on ${who} yet`,
      p.kind === "not_enough_history" ? `The model needs at least ${p.needed} investments before 2012 to learn what an investor likes. ${capitalise(who)} has ${p.pastDeals}.` : "",
      "other", "Choose another investor"],
    bad_request: ["The page asked for something the recommender can't answer", "This is a bug in the page, not your input. Choose another investor and try again.", "other", "Choose another investor"],
  };
  const [title, text, action, label] = screens[p.kind];
  return `<div class="empty" role="alert"><h3>${title}</h3><p>${text}</p>
    <button type="button" class="${action === "retry" ? "primary" : "secondary"}" id="${action}">${label}</button></div>`;
}
