// Pass mark 5 (quality floor): text contrast of at least 4.5:1, checked with the WCAG formula, not by eye.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

// Read the stylesheet as plain text (the test runner does not hand CSS imports to tests).
// Tests run from the frontend folder (npm test), so the path is relative to it.
const css = readFileSync("src/styles.css", "utf8");

const token = (name: string): string => {
  const m = css.match(new RegExp(`--${name}:\\s*(#[0-9A-Fa-f]{6})`));
  if (!m) throw new Error(`token --${name} missing from styles.css`);
  return m[1]!;
};
const channel = (v: number) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const luminance = (hex: string) => {
  const [r, g, b] = [1, 3, 5].map((i) => channel(parseInt(hex.slice(i, i + 2), 16) / 255));
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
};
export const contrast = (a: string, b: string) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
};

describe("palette", () => {
  it("has exactly the six agreed colours", () => {
    expect([token("paper"), token("panel"), token("ink"), token("quiet"), token("line"), token("accent")])
      .toEqual(["#F3F5F2", "#FFFFFF", "#1B201B", "#566056", "#D7DDD5", "#465724"]);
  });

  it.each([
    ["ink on paper", "ink", "paper"], ["ink on panel", "ink", "panel"],
    ["quiet text on paper", "quiet", "paper"], ["quiet text on panel", "quiet", "panel"],
    ["key result line on paper", "accent", "paper"],
  ])("%s is at least 4.5:1", (_, fg, bg) => {
    expect(contrast(token(fg), token(bg))).toBeGreaterThanOrEqual(4.5);
  });

  it("white text on the main button is at least 4.5:1", () => {
    expect(contrast("#FFFFFF", token("accent"))).toBeGreaterThanOrEqual(4.5);
  });

  it("uses the accent only for the main action, the key result line and focus", () => {
    const uses = css.split("\n").filter((l) => l.includes("var(--accent)")).map((l) => l.trim().split(" {")[0]);
    expect(uses).toEqual([":focus-visible", ".primary", ".key"]);
  });
});
