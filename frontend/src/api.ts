// Talks to the Python API and turns every failure into a plain, named problem.
// The page never shows a raw error or a blank screen: each problem has its own screen.

import type { Champion, InvestorSummary, Picks, Profile } from "./types";

export type Problem =
  | { kind: "unreachable" }                                   // no answer at all (API not running, network down)
  | { kind: "unknown_investor" }                              // 404
  | { kind: "not_enough_history"; pastDeals: number; needed: number }
  | { kind: "bad_request"; message: string }                  // other 422s: a bug on our side, said plainly
  | { kind: "server" };                                       // 5xx

export class ApiError extends Error {
  constructor(public readonly problem: Problem) {
    super(problem.kind);
  }
}

type Fetch = (input: string, init?: RequestInit) => Promise<Response>;

export async function getJson<T>(path: string, fetchImpl: Fetch = fetch): Promise<T> {
  let response: Response;
  try {
    response = await fetchImpl(path, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError({ kind: "unreachable" });
  }
  if (response.ok) {
    try {
      return (await response.json()) as T;
    } catch {
      throw new ApiError({ kind: "server" });
    }
  }
  let detail: unknown = null;
  try {
    detail = ((await response.json()) as { detail?: unknown }).detail;
  } catch {
    /* a non-JSON error body still maps to a problem below */
  }
  if (response.status === 404) throw new ApiError({ kind: "unknown_investor" });
  if (response.status === 422) {
    const d = detail as { code?: string; past_deals?: number; needed?: number } | null;
    if (d && d.code === "not_enough_history" && typeof d.past_deals === "number" && typeof d.needed === "number") {
      throw new ApiError({ kind: "not_enough_history", pastDeals: d.past_deals, needed: d.needed });
    }
    throw new ApiError({ kind: "bad_request", message: "The page asked the recommender something it could not answer." });
  }
  if (response.status === 502 || response.status === 503 || response.status === 504) {
    throw new ApiError({ kind: "unreachable" }); // the dev proxy answers 5xx when the API is not running
  }
  throw new ApiError({ kind: "server" });
}

export interface Api {
  champion(): Promise<Champion>;
  search(text: string, limit?: number): Promise<InvestorSummary[]>;
  profile(id: string): Promise<Profile>;
  picks(id: string, k?: number): Promise<Picks>;
}

export function makeApi(base = "/api", fetchImpl: Fetch = (i, init) => fetch(i, init)): Api {
  const q = (params: Record<string, string | number>) => new URLSearchParams(
    Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
  return {
    champion: () => getJson<Champion>(`${base}/champion`, fetchImpl),
    search: async (text, limit = 30) =>
      (await getJson<{ investors: InvestorSummary[] }>(`${base}/investors?${q({ search: text, limit })}`, fetchImpl)).investors,
    profile: (id) => getJson<Profile>(`${base}/investor?${q({ investor: id })}`, fetchImpl),
    picks: (id, k = 10) => getJson<Picks>(`${base}/picks?${q({ investor: id, k })}`, fetchImpl),
  };
}
