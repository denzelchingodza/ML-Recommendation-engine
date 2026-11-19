// The shapes the Python API sends back (backend/recommender/api).

export interface Champion { model: string; version: number }

export interface InvestorSummary { id: string; name: string; past_deals: number; enough_history: boolean }

export interface Profile {
  id: string; name: string; past_deals: number; enough_history: boolean; needed: number; cutoff: string;
  top_state: { code: string; name: string; count: number } | null;
  top_sector: { name: string; count: number } | null;
  later_investments: number | null;
}

export interface Reason {
  label: "state" | "category"; value: string; value_name: string; part: number;
  past_deals: number; examples: string[]; example_names: string[];
}

export type Match = "close" | "partial" | "possible";

export interface Pick {
  company: string; name: string; score: number; why: string; by_chance: boolean;
  match: Match; equally_strong: number; reasons: Reason[]; backed_later: boolean;
}

export interface Picks {
  investor: string; name: string; past_deals: number; champion: Champion; cutoff: string;
  hindsight_available: boolean; picks: Pick[];
}
