// Pass mark 3: every failure becomes a named problem, never a crash or a blank page.
import { describe, expect, it } from "vitest";
import { ApiError, getJson, makeApi } from "./api";
import notEnough from "./fixtures/picks-act.json";
import unknown from "./fixtures/picks-unknown.json";

const reply = (status: number, body: unknown) => async () =>
  new Response(typeof body === "string" ? body : JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const problemOf = async (p: Promise<unknown>) => {
  try { await p; } catch (err) { return (err as ApiError).problem; }
  throw new Error("expected a problem");
};

describe("api problems", () => {
  it("no answer at all is 'unreachable'", async () => {
    expect(await problemOf(getJson("/x", async () => { throw new TypeError("Failed to fetch"); }))).toEqual({ kind: "unreachable" });
  });
  it("the dev proxy's 502/503 when the API is down is also 'unreachable'", async () => {
    expect(await problemOf(getJson("/x", reply(502, "Bad gateway")))).toEqual({ kind: "unreachable" });
  });
  it("a real 404 from the API is 'unknown_investor'", async () => {
    expect(await problemOf(getJson("/x", reply(unknown.status, unknown.body)))).toEqual({ kind: "unknown_investor" });
  });
  it("the real not-enough-history answer keeps its numbers", async () => {
    expect(await problemOf(getJson("/x", reply(notEnough.status, notEnough.body))))
      .toEqual({ kind: "not_enough_history", pastDeals: 3, needed: 5 });
  });
  it("any other 422 is a plain 'bad_request'", async () => {
    expect((await problemOf(getJson("/x", reply(422, { detail: [{ msg: "bad k" }] }))))!.kind).toBe("bad_request");
  });
  it("a 500 or a broken body is 'server'", async () => {
    expect(await problemOf(getJson("/x", reply(500, { detail: "oops" })))).toEqual({ kind: "server" });
    expect(await problemOf(getJson("/x", reply(200, "not json")))).toEqual({ kind: "server" });
  });
  it("builds the right URLs, with ids safely encoded", async () => {
    const seen: string[] = [];
    const api = makeApi("/api", async (url) => { seen.push(url); return new Response("{}", { status: 200 }); });
    await api.picks("/financial-organization/a&b", 10);
    expect(seen[0]).toBe("/api/picks?investor=%2Ffinancial-organization%2Fa%26b&k=10");
  });
});
