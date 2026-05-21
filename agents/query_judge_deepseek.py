"""
DeepSeek / Ollama judge: one focused prompt per query variant.

DeepSeek-R1 often ignores multi-query output constraints; separate calls with a
single-line answer requirement work more reliably.
"""

from __future__ import annotations

import re

from agents.query_judge_agent import (
    JudgeContext,
    _format_ce_rows_for_judge,
    _format_ranked_results_for_judge,
)
from llm_provider import OllamaProvider

VARIANT_SPECS = (
    (
        "legal_terminology_rewrite",
        "Rewrite using formal legal and regulatory terminology for the same business process.",
    ),
    (
        "regulatory_compliance_query",
        "Focus on legal obligations, requirements, restrictions, timelines, reporting, and consumer or data-protection duties.",
    ),
    (
        "contract_clause_query",
        "Focus on contract clauses, policy terms, duties, rights, and procedural provisions.",
    ),
    (
        "risk_scenario_query",
        "Focus on violations, failures, disputes, penalties, remedies, and consequences.",
    ),
)


def build_single_variant_prompt(
    field: str,
    direction: str,
    context: JudgeContext,
    documents,
    current_query: str,
) -> str:
    bm25_context = _format_ranked_results_for_judge(context.bm25_results, documents)
    ce_context = _format_ce_rows_for_judge(context.ce_rows)
    return f"""You improve ONE legal-information-retrieval query for a regulatory text corpus.

Original business-process query:
\"\"\"{context.original_query}\"\"\"

Current {field} (rewrite this):
\"\"\"{current_query}\"\"\"

Variant direction:
{direction}

Top 15 BM25 passages for the original query:
{bm25_context}

Top 15 cross-encoder passages for the original query:
{ce_context}

Task:
Write ONE improved BM25 search query for this variant only. Optimize for recall (include useful legal/regulatory vocabulary from the passages). Stay on the original process topic.

Output rules (strict):
- Reply with ONLY the rewritten query text.
- No JSON, markdown, numbering, labels, bullets, quotes around the whole answer, or explanation.
- Do not include reasoning, thinking tags, or analysis — output the query line only.
"""


def _strip_model_artifacts(text: str) -> str:
    think_block = re.compile(
        "<" + "think" + r">.*?</" + "think" + ">",
        flags=re.DOTALL | re.IGNORECASE,
    )
    redacted_block = re.compile(
        r"<think>.*?</think>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = think_block.sub("", text)
    cleaned = redacted_block.sub("", cleaned)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _answer_section(text: str) -> str:
    match = re.search(r"</think>\s*", text, flags=re.IGNORECASE)
    if match:
        return text[match.end() :].strip()
    match = re.search(r"</" + "think" + r">\s*", text, flags=re.IGNORECASE)
    if match:
        return text[match.end() :].strip()
    return text


_BOILERPLATE_STARTS = (
    "here are",
    "here is",
    "the improved",
    "the refined",
    "based on",
    "output:",
    "answer:",
)


def parse_single_variant_response(response: str) -> str:
    """Extract one query string from a single-variant judge response."""
    text = _strip_model_artifacts(_answer_section(response))
    if not text:
        raise ValueError("Empty judge response")

    candidates: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        quoted = re.search(r'["\'](.+)["\']\s*$', line)
        if quoted:
            candidates.append(quoted.group(1).strip())
            continue

        bullet_quoted = re.match(r'^[-•*]\s*["\'](.+?)["\']\s*$', line)
        if bullet_quoted:
            candidates.append(bullet_quoted.group(1).strip())
            continue

        labeled = re.match(
            r"^(?:\d+\.\s*)?\*\*[^*]+\*\*\s*:\s*[\"']?(.+?)[\"']?\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if labeled:
            candidates.append(labeled.group(1).strip().strip("\"'"))
            continue

        line = re.sub(r"^[-•*]\s*", "", line).strip().strip("\"'")
        lower = line.lower()
        if len(line) < 15:
            continue
        if any(lower.startswith(prefix) for prefix in _BOILERPLATE_STARTS):
            continue
        if line.endswith(":"):
            continue
        candidates.append(line)

    if candidates:
        return candidates[-1]

    one_line = " ".join(text.split())
    if len(one_line) >= 15:
        return one_line.strip("\"'")

    raise ValueError("No query line found in judge response")


def deepseek_judge_refine_queries(
    context: JudgeContext,
    documents,
    *,
    model: str,
    base_url: str,
    recorded_value,
    query_variants_cls,
):
    """Run four separate judge calls (one per variant) and assemble QueryVariants."""
    llm = OllamaProvider(model=model, base_url=base_url)
    fallback = context.variants
    refined: dict[str, str] = {}

    for field, direction in VARIANT_SPECS:
        current = getattr(fallback, field)
        prompt = build_single_variant_prompt(
            field, direction, context, documents, current
        )
        response = llm.generate(prompt)

        print(f"[DeepSeek judge / {field}] raw model output:")
        print(response)
        print("-" * 60)

        try:
            query = parse_single_variant_response(response)
            refined[field] = recorded_value(query, current)
        except Exception as exc:
            print(
                f"[WARN] Failed to parse DeepSeek judge response for {field} ({exc}); "
                "keeping current variant."
            )
            refined[field] = current

    return query_variants_cls(
        legal_terminology_rewrite=refined["legal_terminology_rewrite"],
        regulatory_compliance_query=refined["regulatory_compliance_query"],
        contract_clause_query=refined["contract_clause_query"],
        risk_scenario_query=refined["risk_scenario_query"],
    )
