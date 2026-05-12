
import time
from dataclasses import dataclass

import pandas as pd

from agents.clause_contract_agent import clause_contract_agent
from agents.legal_terminology_rewriter import legal_terminology_rewriter
from agents.regulatory_compliance_agent import regulatory_compliance_agent
from agents.scenario_risk_agent import scenario_risk_agent
from retrieval.sota_retrieval import RetrievedPassage, SotaRetriever, load_corpus_texts


# ----------------------------
# Config
# ----------------------------
INPUT_PATH = "regulatory_relevance4process-D73C/SOTA_NLP_LIR/input_ranking/uc1/Input_queries_medium_uc1.xlsx"
CORPUS_PATH = "regulatory_relevance4process-D73C/SOTA_NLP_LIR/input_ranking/uc1/Input_corpus_uc1.xlsx"
OUTPUT_PATH = "output_with_agents.csv"
OUTPUT_RANKING_PATH = "output_ranking.xlsx"
LLM_PROVIDER = "gemini"  # "ollama" or "gemini"
NUM_ROWS = None
TOP_K = 30
SLEEP_SECONDS = 1


# ----------------------------
@dataclass
class QueryVariants:
    legal_terminology_rewrite: str
    regulatory_compliance_query: str
    contract_clause_query: str
    risk_scenario_query: str


# ----------------------------
# Prompt + Generation
# ----------------------------
def clean_text(text: str) -> str:
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


def generate_questions(text: str, provider: str = LLM_PROVIDER) -> QueryVariants:
    """Call all 4 agents to generate different query perspectives"""

    legal_term = legal_terminology_rewriter(text, provider=provider)
    regulatory = regulatory_compliance_agent(text, provider=provider)
    contract_clause = clause_contract_agent(text, provider=provider)
    risk_scenario = scenario_risk_agent(text, provider=provider)

    return QueryVariants(
        legal_terminology_rewrite=legal_term,
        regulatory_compliance_query=regulatory,
        contract_clause_query=contract_clause,
        risk_scenario_query=risk_scenario,
    )


def compose_extended_query(original_query: str, variants: QueryVariants) -> str:
    parts = [
        original_query,
        variants.legal_terminology_rewrite,
        variants.regulatory_compliance_query,
        variants.contract_clause_query,
        variants.risk_scenario_query,
    ]
    return " ".join(clean_text(part) for part in parts if str(part).strip())


def retrieval_rows_to_df(rows: list[RetrievedPassage]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query": row.query,
                "retrieval_query": row.retrieval_query,
                "rel_text": row.rel_text,
                "score": row.score,
                "method": row.method,
                "query_variant": row.query_variant,
            }
            for row in rows
        ]
    )


# ----------------------------
# Main Pipeline
# ----------------------------
def retrieval_pipeline(provider: str = LLM_PROVIDER):
    df = pd.read_excel(INPUT_PATH)

    if "process_text" not in df.columns:
        raise ValueError("Column 'process_text' not found")

    if NUM_ROWS is not None:
        df = df.head(NUM_ROWS).copy()
    else:
        df = df.copy()

    df["process_text"] = df["process_text"].apply(clean_text)

    results = {
        "legal_terminology_rewrite": [],
        "regulatory_compliance_query": [],
        "contract_clause_query": [],
        "risk_scenario_query": [],
        "extended_query": [],
    }

    corpus_texts = load_corpus_texts(CORPUS_PATH)
    retriever = SotaRetriever(corpus_texts=corpus_texts)

    all_bm25_ce_baseline: list[RetrievedPassage] = []
    all_bi_ce_baseline: list[RetrievedPassage] = []
    all_bm25_ce_extended: list[RetrievedPassage] = []
    all_bi_ce_extended: list[RetrievedPassage] = []

    for idx, text in enumerate(df["process_text"], 1):
        print(f"Processing row {idx}/{len(df)}...")
        agent_results = generate_questions(text, provider=provider)
        extended_query = compose_extended_query(text, agent_results)

        results["legal_terminology_rewrite"].append(agent_results.legal_terminology_rewrite)
        results["regulatory_compliance_query"].append(agent_results.regulatory_compliance_query)
        results["contract_clause_query"].append(agent_results.contract_clause_query)
        results["risk_scenario_query"].append(agent_results.risk_scenario_query)
        results["extended_query"].append(extended_query)

        all_bm25_ce_baseline.extend(
            retriever.search_bm25_ce(TOP_K, text, query_variant="baseline", result_query=text)
        )
        all_bi_ce_baseline.extend(
            retriever.search_bi_ce(TOP_K, text, query_variant="baseline", result_query=text)
        )
        all_bm25_ce_extended.extend(
            retriever.search_bm25_ce(
                TOP_K, extended_query, query_variant="query_extension", result_query=text
            )
        )
        all_bi_ce_extended.extend(
            retriever.search_bi_ce(
                TOP_K, extended_query, query_variant="query_extension", result_query=text
            )
        )

        time.sleep(SLEEP_SECONDS)  # rate limiting

    for key, values in results.items():
        df[key] = values

    df.to_csv(OUTPUT_PATH, index=False)

    df_bm25_baseline = retrieval_rows_to_df(all_bm25_ce_baseline)
    df_bi_baseline = retrieval_rows_to_df(all_bi_ce_baseline)
    df_bm25_extended = retrieval_rows_to_df(all_bm25_ce_extended)
    df_bi_extended = retrieval_rows_to_df(all_bi_ce_extended)

    with pd.ExcelWriter(OUTPUT_RANKING_PATH) as writer:
        df_bm25_baseline.to_excel(writer, sheet_name="BM25_CE_baseline", index=False)
        df_bi_baseline.to_excel(writer, sheet_name="Bi_CE_baseline", index=False)
        df_bm25_extended.to_excel(writer, sheet_name="BM25_CE_query_extension", index=False)
        df_bi_extended.to_excel(writer, sheet_name="Bi_CE_query_extension", index=False)

    print(f"✅ Done. Output saved to: {OUTPUT_PATH} and {OUTPUT_RANKING_PATH}")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    retrieval_pipeline()