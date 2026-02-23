/**
 * Research run estimates from ReportConfig-style params.
 * Formula aligns with README: Planner (1) + per-node (tool_plan + tool runs + synthesis + gaps) + Writer (1).
 */

export interface ResearchParams {
  num_plan_steps: number;
  max_depth: number;
  max_probes: number;
  max_tool_pairs: number;
}

/** Upper-bound executed nodes: roots + children up to max_depth (simplified). */
function estimateMaxNodes(p: ResearchParams): number {
  let nodes = p.num_plan_steps;
  let level = p.num_plan_steps;
  for (let d = 1; d < p.max_depth; d++) {
    level = level * p.max_probes;
    nodes += level;
    if (nodes > 200) return 200;
  }
  return Math.min(nodes, 200);
}

/** Estimated LLM API calls (Planner + per-node * (tool_plan + synthesis + gaps) + Writer). */
export function estimateLLMCalls(p: ResearchParams): number {
  const nodes = estimateMaxNodes(p);
  const perNode = 1 + p.max_tool_pairs + 2; // tool plan + tool runs (each can be 1 call) + synthesis + gaps
  return 1 + nodes * perNode + 1;
}

/** Estimated tool/API calls (search, scrape, etc.) — up to max_tool_pairs per node. */
export function estimateToolCalls(p: ResearchParams): number {
  const nodes = estimateMaxNodes(p);
  return nodes * p.max_tool_pairs;
}

/** Rough cost estimate: Gemini Flash ~$0.075/1M input, ~$0.30/1M output; assume ~2k tokens/call. */
const ESTIMATE_TOKENS_PER_LLM_CALL = 2500;
const ESTIMATE_INPUT_RATIO = 0.7;
const GEMINI_FLASH_INPUT_PER_M = 0.075;
const GEMINI_FLASH_OUTPUT_PER_M = 0.30;

export function estimateCostUSD(p: ResearchParams, llmCalls?: number): number {
  const calls = llmCalls ?? estimateLLMCalls(p);
  const totalTokens = calls * ESTIMATE_TOKENS_PER_LLM_CALL;
  const inputTokens = totalTokens * ESTIMATE_INPUT_RATIO;
  const outputTokens = totalTokens * (1 - ESTIMATE_INPUT_RATIO);
  const inputCost = (inputTokens / 1_000_000) * GEMINI_FLASH_INPUT_PER_M;
  const outputCost = (outputTokens / 1_000_000) * GEMINI_FLASH_OUTPUT_PER_M;
  return Math.round((inputCost + outputCost) * 100) / 100;
}

export const DEFAULT_RESEARCH_PARAMS: ResearchParams = {
  num_plan_steps: 3,
  max_depth: 2,
  max_probes: 3,
  max_tool_pairs: 3,
};
