import { describe, expect, test, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, readFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  segmentClaims,
  selectJudgeModel,
  resolveProvider,
  parseVerdict,
  buildNudge,
  PremiseStore,
  Judge,
  makeCache,
  verdictKey,
  extractPackages,
} from "../src/index";

const GLM = "hf:zai-org/GLM-5.3-Flash";
const DS = "hf:deepseek-ai/DeepSeek-V4.1-Flash";

function fakeFetch(body: unknown, status = 200): typeof fetch {
  const resp = {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
  return (async () => resp) as unknown as typeof fetch;
}

function okResp(verdict: string, fenced = false) {
  const content = fenced
    ? `\`\`\`json\n{"verdict": "${verdict}"}\n\`\`\``
    : `{"verdict": "${verdict}"}`;
  return {
    choices: [{ message: { content } }],
    usage: { prompt_tokens: 10, completion_tokens: 5 },
  };
}

function noSleep(): Promise<void> {
  return Promise.resolve();
}

describe("claim segmentation", () => {
  test("3-sentence EN reply -> 3 claims", () => {
    const claims = segmentClaims(
      "The sky is blue. Water boils at 100 degrees. Paris is in France.",
    );
    expect(claims.length).toBe(3);
  });

  test("3-sentence DE reply -> 3 claims", () => {
    const claims = segmentClaims(
      "Der Himmel ist blau. Wasser kocht bei 100 Grad. Paris liegt in Frankreich.",
    );
    expect(claims.length).toBe(3);
  });

  test("empty/whitespace drops out", () => {
    expect(segmentClaims("   ").length).toBe(0);
  });
});

describe("judge selection", () => {
  const cfg = resolveProvider({
    RFE_JUDGE_PROVIDER: "synthetic",
  } as Record<string, string>);

  test("glm-flash active -> deepseek judge", () => {
    expect(selectJudgeModel(GLM, cfg)).toBe(DS);
    expect(selectJudgeModel("z-ai/glm-5.3-flash", cfg)).toBe(DS);
  });

  test("anything else -> glm judge", () => {
    expect(selectJudgeModel("anthropic/claude-sonnet-4", cfg)).toBe(GLM);
    expect(selectJudgeModel("openai/gpt-5", cfg)).toBe(GLM);
    expect(selectJudgeModel("unknown", cfg)).toBe(GLM);
  });

  test("self-judge impossible when active model deliberately matches", () => {
    const judge = selectJudgeModel(GLM, cfg);
    expect(judge).not.toBe(GLM);
    expect(judge.toLowerCase()).toContain("deepseek");
    // and if the deepseek judge itself were active, judge flips back to glm
    expect(selectJudgeModel(DS, cfg)).toBe(GLM);
  });

  test("env overrides win", () => {
    const c = resolveProvider({
      RFE_JUDGE_PROVIDER: "openrouter",
      RFE_JUDGE_DS_MODEL: "custom/ds",
      RFE_JUDGE_GLM_MODEL: "custom/glm",
    } as Record<string, string>);
    expect(c.glmModel).toBe("custom/glm");
    expect(c.deepseekModel).toBe("custom/ds");
    expect(c.baseUrl).toBe("https://openrouter.ai/api/v1");
  });
});

describe("verdict parsing", () => {
  test("valid JSON", () => {
    expect(parseVerdict('{"verdict": "faithful"}')).toBe("faithful");
  });

  test("markdown-fenced JSON", () => {
    expect(parseVerdict('```json\n{"verdict": "unfaithful"}\n```')).toBe(
      "unfaithful",
    );
  });

  test("garbage -> throw (judge falls back to unverifiable + parseErrors++)", async () => {
    expect(() => parseVerdict("total garbage")).toThrow();
    let calls = 0;
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      fetchImpl: (async () => {
        calls++;
        return {
          ok: true,
          status: 200,
          json: async () => okResp("nonsense"),
        } as Response;
      }) as unknown as typeof fetch,
      sleepImpl: noSleep,
    });
    const v = await judge.verdict("ctx", "claim");
    expect(v).toBe("unverifiable");
    expect(judge.parseErrors).toBe(1);
    expect(calls).toBe(2); // retry once with 2x max_tokens on parse failure
  });
});

describe("cache", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "rfe-cache-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    delete process.env["RFE_CACHE_DIR"];
  });

  test("same (model,context,claim) twice -> one fetch call", async () => {
    let calls = 0;
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      cache: makeCache(GLM, {}),
      fetchImpl: (async () => {
        calls++;
        return { ok: true, status: 200, json: async () => okResp("faithful") } as Response;
      }) as unknown as typeof fetch,
      sleepImpl: noSleep,
    });
    expect(await judge.verdict("c", "p")).toBe("faithful");
    expect(await judge.verdict("c", "p")).toBe("faithful");
    expect(calls).toBe(1);
  });

  test("RFE_CACHE_DIR set -> file written", async () => {
    process.env["RFE_CACHE_DIR"] = dir;
    const judge = new Judge({
      baseUrl: "https://x.test",
      apiKey: "k",
      model: GLM,
      session: "s",
      cache: makeCache(GLM),
      fetchImpl: fakeFetch(okResp("faithful")),
      sleepImpl: noSleep,
    });
    await judge.verdict("ctx", "claim");
    const file = join(dir, "opencode-cache-hf_zai-org_GLM-5.3-Flash.jsonl");
    expect(existsSync(file)).toBe(true);
    const row = JSON.parse(readFileSync(file, "utf8").trim());
    expect(row.key).toBe(verdictKey(GLM, "ctx", "claim"));
    expect(row.verdict).toBe("faithful");
  });

  test("unset -> memory only", async () => {
    delete process.env["RFE_CACHE_DIR"];
    const cache = makeCache(GLM);
    const key = verdictKey(GLM, "a", "b");
    cache.store(key, "faithful");
    expect(cache.get(key)).toBe("faithful");
    expect(existsSync(join(dir, "nothing.jsonl"))).toBe(false);
  });
});

describe("premise cap", () => {
  test("30k chars in -> most recent 24k retained", () => {
    const p = new PremiseStore(24_000);
    p.append("x".repeat(30_000));
    expect(p.length).toBe(24_000);
    p.append("A".repeat(10_000));
    p.append("B".repeat(20_000));
    expect(p.length).toBe(24_000);
    expect(p.text.startsWith("A".repeat(4000 - 1))).toBe(true); // older tail kept
    expect(p.text.endsWith("B".repeat(20_000))).toBe(true); // newest kept
  });
});

describe("nudge aggregation", () => {
  test("2 flagged claims -> exactly one nudge message", () => {
    const nudge = buildNudge(GLM, [
      { claim: "Sky is green.", verdict: "unfaithful" },
      { claim: "Moon is cheese.", verdict: "unverifiable" },
    ]);
    expect(nudge).toContain("2 claim(s)");
    expect(nudge).toContain(GLM);
    expect(nudge).toContain("unfaithful (1)");
    expect(nudge).toContain("unverifiable (1)");
    expect(nudge).toContain("do not invent corrections");
    expect((nudge.match(/ragfaith judge/g) ?? []).length).toBe(1);
  });
});

describe("doc-pull heuristic", () => {
  test("install args captured", () => {
    const pkgs = extractPackages("bun install lodash @types/node -D");
    expect(pkgs.has("lodash")).toBe(true);
    expect(pkgs.has("@types/node")).toBe(true);
  });

  test("imports captured", () => {
    const pkgs = extractPackages(`import x from "express"; const y = require("zod");`);
    expect(pkgs.has("express")).toBe(true);
    expect(pkgs.has("zod")).toBe(true);
  });
});

describe("hooks never throw", () => {
  let stderrLines: string[];

  test("judge fetch rejects -> unverifiable, no exception, log emitted", async () => {
    const origWrite = process.stderr.write.bind(process.stderr);
    stderrLines = [];
    (process.stderr as { write: unknown }).write = (chunk: unknown) => {
      stderrLines.push(String(chunk));
      return true;
    };
    try {
      const judge = new Judge({
        baseUrl: "https://x.test",
        apiKey: "k",
        model: GLM,
        session: "s",
        fetchImpl: (async () => {
          throw new Error("network down");
        }) as unknown as typeof fetch,
        sleepImpl: noSleep,
      });
      const v = await judge.verdict("ctx", "claim");
      expect(v).toBe("unverifiable");
      const errLog = stderrLines.find((l) => l.includes('"kind":"error"'));
      expect(errLog).toBeDefined();
    } finally {
      (process.stderr as { write: unknown }).write = origWrite;
    }
  });
});
