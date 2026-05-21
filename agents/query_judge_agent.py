"""
Query Judge Agent
Refines the four agent query variants using top BM25 and cross-encoder retrieval context.
"""

import json
import re
from dataclasses import dataclass

import pandas as pd

from llm_provider import get_llm_provider
from retrieval.retrieval_bm25 import RankingResult


@dataclass
class JudgeContext:
    original_query: str
    variants: object
    bm25_results: list[RankingResult]
    ce_rows: pd.DataFrame


@dataclass
class JudgeRefinedVariants:
    legal_terminology_rewrite: str
    regulatory_compliance_query: str
    contract_clause_query: str
    risk_scenario_query: str


def _clean_text(text: str) -> str:
    cleaned = str(text)
    cleaned = cleaned.replace("or\n\n\n", " ")
    cleaned = cleaned.replace("or\n\n", " ")
    cleaned = cleaned.replace("and\n\n\n", " ")
    cleaned = cleaned.replace("and\n\n", " ")
    cleaned = cleaned.replace("\n\n\n", " ")
    cleaned = cleaned.replace("\n\n", " ")
    cleaned = cleaned.replace("\n \n", " ")
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def _recorded_value(value, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    if str(value).strip() == "":
        return fallback
    return _clean_text(value)


def _truncate_text(text: str, max_chars: int = 900) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _format_ranked_results_for_judge(
    results: list[RankingResult],
    documents,
    limit: int = 15,
) -> str:
    lines = []
    for result in results[:limit]:
        text = _truncate_text(documents[result.document.doc_id].text)
        lines.append(f"{result.rank}. score={result.score:.4f} text={text}")
    return "\n".join(lines)


def _format_ce_rows_for_judge(df: pd.DataFrame, limit: int = 15) -> str:
    if df.empty:
        return ""
    rows = df.sort_values("score", ascending=False).head(limit)
    lines = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        lines.append(f"{rank}. score={float(row['score']):.4f} text={_truncate_text(row['rel_text'])}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_prompt(context: JudgeContext, documents) -> str:
    bm25_context = _format_ranked_results_for_judge(context.bm25_results, documents)
    ce_context = _format_ce_rows_for_judge(context.ce_rows)
    variants = context.variants
    return f"""
You are a legal information retrieval query judge.

Goal:
Improve recall for a regulatory corpus while keeping each query variant's direction.

You receive:
1. The original business-process query.
2. Four current query variants.
3. The top 15 BM25 results for the original query.
4. The top 15 cross-encoder reranked results.

Task:
Adjust the four query variants so the next BM25 retrieval pass is more likely to recover relevant regulatory passages.
Use the result samples to infer corpus vocabulary, legal phrasing, and missing concepts.
Do not make the variants identical. Preserve their direction:
- legal_terminology_rewrite: formal legal terminology for the same process.
- regulatory_compliance_query: obligations, requirements, restrictions, timelines, reporting, consumer/data protection duties.
- contract_clause_query: clauses, duties, rights, policy terms, procedural provisions.
- risk_scenario_query: violations, failures, disputes, penalties, remedies, consequences.

Rules:
- Optimize for high recall, not concise wording.
- Keep each query related to the original process.
- Do not quote result texts verbatim unless useful terms are needed.
- Return ONLY valid JSON with exactly these keys:
  "legal_terminology_rewrite", "regulatory_compliance_query", "contract_clause_query", "risk_scenario_query"

Original query:
\"\"\"{context.original_query}\"\"\"

Current variants:
{{
  "legal_terminology_rewrite": {json.dumps(variants.legal_terminology_rewrite)},
  "regulatory_compliance_query": {json.dumps(variants.regulatory_compliance_query)},
  "contract_clause_query": {json.dumps(variants.contract_clause_query)},
  "risk_scenario_query": {json.dumps(variants.risk_scenario_query)}
}}

Top 15 BM25 results:
{bm25_context}

Top 15 cross-encoder results:
{ce_context}
"""


def query_judge_agent(
    context: JudgeContext,
    documents,
    provider: str = "ollama",
) -> JudgeRefinedVariants:
    """Refines the four query variants using retrieval context from BM25 and cross-encoder results."""
    prompt = build_prompt(context, documents)
    llm = get_llm_provider(provider)
    response = llm.generate(prompt)
    if not response.strip():
        print("[WARN] LLM judge returned empty response; keeping original query variants.")
        return JudgeRefinedVariants(
            legal_terminology_rewrite=context.variants.legal_terminology_rewrite,
            regulatory_compliance_query=context.variants.regulatory_compliance_query,
            contract_clause_query=context.variants.contract_clause_query,
            risk_scenario_query=context.variants.risk_scenario_query,
        )

    try:
        parsed = _extract_json_object(response)
        return JudgeRefinedVariants(
            legal_terminology_rewrite=_recorded_value(
                parsed.get("legal_terminology_rewrite"),
                context.variants.legal_terminology_rewrite,
            ),
            regulatory_compliance_query=_recorded_value(
                parsed.get("regulatory_compliance_query"),
                context.variants.regulatory_compliance_query,
            ),
            contract_clause_query=_recorded_value(
                parsed.get("contract_clause_query"),
                context.variants.contract_clause_query,
            ),
            risk_scenario_query=_recorded_value(
                parsed.get("risk_scenario_query"),
                context.variants.risk_scenario_query,
            ),
        )
    except Exception as exc:
        print(f"[WARN] Failed to parse LLM judge response: {exc}. Keeping original query variants.")
        return JudgeRefinedVariants(
            legal_terminology_rewrite=context.variants.legal_terminology_rewrite,
            regulatory_compliance_query=context.variants.regulatory_compliance_query,
            contract_clause_query=context.variants.contract_clause_query,
            risk_scenario_query=context.variants.risk_scenario_query,
        )


def llm_judge_refine_queries(
    context: JudgeContext,
    documents,
    judge_provider: str,
) -> JudgeRefinedVariants:
    """Backward-compatible alias for query_judge_agent."""
    return query_judge_agent(context, documents, provider=judge_provider)
