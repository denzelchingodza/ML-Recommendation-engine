// Turns API answers into screens. Pure functions returning HTML, so every screen can be tested.
// Design rules (agreed 19 Nov 2026): plain states instead of scores; every claim shows its evidence;
// uncertainty is its own screen; numbers only behind "How we worked this out". All API text is escaped.

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
  <p class="note">${facts ? `${facts}. ` : ""}We'll look for new startups like that.</p>
  <button type="button" class="primary" id="find">Find startups for this investor</button>`;
}

export function matchWords(p: Pick): string {
  if (p.match === "close") return "Same sector and state";
  if (p.match === "partial") return p.reasons[0]?.label === "state" ? "Same state only" : "Same sector only";
  return "Possible";
}

const evidence = (p: Pick): string => p.reasons.length
  ? p.reasons.map((r) => capitalise(r.value_name)).sort((a, b) => a.localeCompare(b)).join(", ")
  : "Nothing in common on record";

const TICK = '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2"/></svg>';

function rows(list: Pick[], hindsight: boolean): string {
  return `<ol class="results">${list.map((p) => `<li><span class="name">${e(p.name)}</span><span class="state">${matchWords(p)}</span>
    <span class="why">${e(evidence(p))}</span>${hindsight && p.backed_later ? `<span class="later">${TICK}They invested</span>` : "<span></span>"}</li>`).join("")}</ol>`;
}

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

function howPanel(data: Picks, list: Pick[], profile: Profile | null): string {
  const y = year(data.cutoff);
  const groups = new Map<number, Pick[]>();
  for (const p of list) groups.set(p.score, [...(groups.get(p.score) ?? []), p]);
  const parts = [...groups.values()].map((g) => {
    const p = g[0]!, shown = g.length, labels = listOf(p.reasons.map((r) => r.value_name));
    const examples = listOf([...new Set(p.reasons.flatMap((r) => r.example_names))]);
    const scoreParts = p.reasons.map((r) => `${e(r.value_name)} gives ${r.part.toFixed(2)}`).join(", ");
    return `<p>${p.equally_strong > shown
        ? `${p.equally_strong} startups match exactly as well as ${shown === 1 ? "this one" : `these ${shown}`} (${e(labels)}). We can't tell them apart, so we show ${shown} of them at random.`
        : `No other startup matches as well as ${shown === 1 ? "this one" : `these ${shown}`} (${e(labels)}).`}</p>
      ${examples ? `<p>Past investments they look like: ${e(examples)}.</p>` : ""}
      <p>For the curious: ${shown === 1 ? "it scores" : "each one scores"} ${p.score.toFixed(2)} out of 1${scoreParts ? ` (${scoreParts})` : ""}.</p>`;
  }).join("");
  const deals = profile ? profile.past_deals : data.past_deals;
  return `<div class="how" id="how-panel"><p>We compared every startup that raised money from ${y} on with ${e(possessive(data.name))} ${deals} past investments, on two things: sector and state.</p>${parts}</div>`;
}

export interface ResultsOptions { hindsight: boolean; how: boolean }

export function results(data: Picks, profile: Profile | null, opts: ResultsOptions): string {
  const list = confident(data.picks);
  if (!list.length) {
    return `<div class="empty"><h3>No close matches for ${e(data.name)}</h3>
      <p>Startups we're less sure about are in the Review tab.</p>
      <button type="button" class="secondary" id="to-review">Go to Review</button></div>`;
  }
  const y = year(data.cutoff), n = list.length, hits = list.filter((p) => p.backed_later).length;
  const later = profile?.later_investments;
  const outcome = hits ? `${e(data.name)} invested in ${hits} of ${n === 1 ? "this one" : `these ${n}`}.`
    : `${e(data.name)} invested in none of ${n === 1 ? "this one" : `these ${n}`}.${later ? ` It made ${plural(later, "other investment", "other investments")} in that time.` : ""}`;
  return `<p class="key">${keyLine(data, list)}</p>
  <button type="button" class="link" id="how" aria-expanded="${opts.how}" aria-controls="how-panel">How we worked this out</button>
  ${opts.how ? howPanel(data, list, profile) : ""}
  ${data.hindsight_available ? `<section class="did" aria-labelledby="did-title"><h3 id="did-title">Did it work?</h3>
    <p>We only used investments up to ${y}. See what really happened from ${y} on.</p>
    <button type="button" class="secondary" id="hindsight" aria-pressed="${opts.hindsight}">${opts.hindsight ? "Hide what really happened" : "Check against what really happened"}</button>
    ${opts.hindsight ? `<p class="result" role="status">${outcome}</p>` : ""}</section>` : ""}
  ${rows(list, opts.hindsight)}`;
}

export function review(data: Picks, opts: ResultsOptions): string {
  const list = unsure(data.picks);
  if (!list.length) {
    return `<div class="empty"><h3>Nothing to double-check</h3>
      <p>All of these startups match on sector, state or both. Startups we're less sure about, for example ones missing their sector or state, would show up here.</p>
      <button type="button" class="secondary" id="to-results">Back to results</button></div>`;
  }
  return `<p class="key">${plural(list.length, "startup needs", "startups need")} a second look</p>
    <p class="note">We're less sure about these: they're missing a sector or state on record, or share little with ${e(possessive(data.name))} past investments.</p>
    ${rows(list, opts.hindsight)}`;
}

export function problem(p: Problem, investorName: string | null): string {
  const who = investorName ? e(investorName) : "this investor";
  const screens: Record<Problem["kind"], [string, string, string, string]> = {
    unreachable: ["Couldn't load startups right now", "The recommender didn't answer. Check it is running, then try again.", "retry", "Try again"],
    server: ["The recommender ran into a problem", "It has been logged on the recommender's side. Try again in a moment.", "retry", "Try again"],
    unknown_investor: ["We couldn't find that investor", "They may have no investments before 2012 in our data. Choose another investor.", "other", "Choose another investor"],
    not_enough_history: [`Not enough history on ${who} yet`,
      p.kind === "not_enough_history" ? `We need at least ${p.needed} investments before 2012 to see what an investor likes. ${capitalise(who)} has ${p.pastDeals}.` : "",
      "other", "Choose another investor"],
    bad_request: ["The page asked for something the recommender can't answer", "This is a bug in the page, not your input. Choose another investor and try again.", "other", "Choose another investor"],
  };
  const [title, text, action, label] = screens[p.kind];
  return `<div class="empty" role="alert"><h3>${title}</h3><p>${text}</p>
    <button type="button" class="${action === "retry" ? "primary" : "secondary"}" id="${action}">${label}</button></div>`;
}
