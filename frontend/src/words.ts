// Small helpers so sentences read naturally.

export const escapeHtml = (text: string): string =>
  text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

export const capitalise = (text: string): string => (text ? text[0]!.toUpperCase() + text.slice(1) : text);

/** "Austin Ventures" -> "Austin Ventures'", "SV Angel" -> "SV Angel's" */
export const possessive = (name: string): string => (name.endsWith("s") ? `${name}'` : `${name}'s`);

export const plural = (n: number, one: string, many: string): string => `${n} ${n === 1 ? one : many}`;

/** "Adometry", "Alereon", "ACTIVE Network" -> "Adometry, Alereon and ACTIVE Network" */
export const listOf = (items: string[]): string =>
  items.length <= 1 ? (items[0] ?? "") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
