import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agents.clause_contract_agent import clause_contract_agent
from agents.legal_terminology_rewriter import legal_terminology_rewriter
from agents.query_judge_agent import JudgeContext, llm_judge_refine_queries
from agents.regulatory_compliance_agent import regulatory_compliance_agent
from agents.scenario_risk_agent import scenario_risk_agent

VARIANT_AGENT_COLUMNS = [
    "legal_terminology_rewrite",
    "regulatory_compliance_query",
    "contract_clause_query",
    "risk_scenario_query",
]
from retrieval.query_merging import RankedList, ReciprocalRankFusion
from retrieval.retrieval_bm25 import Query, RankingResult, build_bm25_index, load_corpus
from retrieval.sota_retrieval import RetrievedPassage, SotaRetriever

SOTA_BASE = Path("regulatory_relevance4process-D73C/SOTA_NLP_LIR")
INPUT_RANKING = SOTA_BASE / "input_ranking"
OUTPUT_EVAL = SOTA_BASE / "output_ranking_input_eval"


def _algo_output_dir(use_case: str) -> Path:
    return OUTPUT_EVAL / use_case / "algo_output"


def get_level_configs(use_case: str) -> dict:
    """Process / subprocess / task: paths and top-k aligned with original SOTA notebook."""
    uc = use_case.lower()
    if uc not in {"uc1", "uc2", "uc3"}:
        raise ValueError(f"use_case must be 'uc1', 'uc2', or 'uc3', got {use_case}")
    gs = OUTPUT_EVAL / uc / "gold_standard"
    ir = INPUT_RANKING / uc
    return {
        "process": {
            "query_path": None,
            "gs_path": str(gs / f"gs_{uc}_process_level.xlsx"),
            "top_k": 100,
        },
        "subprocess": {
            "query_path": str(ir / f"Input_queries_medium_{uc}.xlsx"),
            "gs_path": str(gs / f"gs_{uc}_subprocess_level.xlsx"),
            "top_k": 30,
        },
        "task": {
            "query_path": str(ir / f"Input_queries_low_{uc}.xlsx"),
            "gs_path": str(gs / f"gs_{uc}_event_level.xlsx"),
            "top_k": 15,
        },
    }


def corpus_path_for(use_case: str) -> str:
    uc = use_case.lower()
    return str(INPUT_RANKING / uc / f"Input_corpus_{uc}.xlsx")


LLM_PROVIDER = "ollama"  # "ollama", "gemini", or "records"
NUM_ROWS = None
SLEEP_SECONDS = 1

LEVEL_TO_ALGO_SUFFIX = {
    "process": "process_level",
    "subprocess": "subprocess_level",
    "task": "event_level",
}

# Weighted RRF: original query + four agents (order must match ranked_lists_weighted)
WEIGHTED_RRF_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]

OUTPUT_SHEETS = {
    "bm25_baseline": "BM25_baseline",
    "bm25_rrf": "BM25_RRF_query_merge",
    "bm25_rrf_weighted": "BM25_RRF_weighted",
    "bm25_ce": "BM25_CE",
    "bm25_rrf_ce": "BM25_RRF_CE",
    "bm25_rrf_weighted_ce": "BM25_RRF_weighted_CE",
    "bm25_rrf_weighted_judged": "BM25_RRF_weighted_judged",
    "bm25_rrf_weighted_ce_judged": "BM25_RRF_weighted_CE_judged",
}


@dataclass
class QueryVariants:
    legal_terminology_rewrite: str
    regulatory_compliance_query: str
    contract_clause_query: str
    risk_scenario_query: str


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


def recorded_queries_path_for(use_case: str) -> Path:
    return Path(f"output_with_agents_{use_case.lower()}.csv")


def generate_variant_field(text: str, column: str, provider: str) -> str:
    if column == "legal_terminology_rewrite":
        return legal_terminology_rewriter(text, provider=provider)
    if column == "regulatory_compliance_query":
        return regulatory_compliance_agent(text, provider=provider)
    if column == "contract_clause_query":
        return clause_contract_agent(text, provider=provider)
    if column == "risk_scenario_query":
        return scenario_risk_agent(text, provider=provider)
    raise ValueError(f"Unknown variant column: {column}")


def generate_questions(text: str, provider: str = LLM_PROVIDER) -> QueryVariants:
    if provider == "records":
        raise ValueError(
            "provider='records' must be handled through query_variants_for(). "
            "Restart the notebook kernel or reload main before rerunning retrieval_pipeline()."
        )

    return QueryVariants(
        legal_terminology_rewrite=generate_variant_field(text, "legal_terminology_rewrite", provider),
        regulatory_compliance_query=generate_variant_field(text, "regulatory_compliance_query", provider),
        contract_clause_query=generate_variant_field(text, "contract_clause_query", provider),
        risk_scenario_query=generate_variant_field(text, "risk_scenario_query", provider),
    )


def variant_field_value(row: dict | None, column: str) -> str | None:
    if row is None or column not in row:
        return None
    value = row[column]
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def missing_variant_columns(query_text: str, row: dict | None) -> list[str]:
    query_clean = clean_text(query_text)
    missing: list[str] = []
    for column in VARIANT_AGENT_COLUMNS:
        value = variant_field_value(row, column)
        if not value:
            missing.append(column)
            continue
        if clean_text(value) == query_clean:
            missing.append(column)

    if not missing:
        return []

    if len(missing) == len(VARIANT_AGENT_COLUMNS):
        return missing

    return missing


def variants_from_row(row: dict | None, query_text: str) -> QueryVariants:
    query_clean = clean_text(query_text)
    return QueryVariants(
        legal_terminology_rewrite=recorded_value(
            variant_field_value(row, "legal_terminology_rewrite"), query_clean
        ),
        regulatory_compliance_query=recorded_value(
            variant_field_value(row, "regulatory_compliance_query"), query_clean
        ),
        contract_clause_query=recorded_value(
            variant_field_value(row, "contract_clause_query"), query_clean
        ),
        risk_scenario_query=recorded_value(
            variant_field_value(row, "risk_scenario_query"), query_clean
        ),
    )


def generate_missing_variants(
    text: str,
    row: dict | None,
    provider: str,
    columns: list[str] | None = None,
) -> QueryVariants:
    columns = columns or missing_variant_columns(text, row)
    current = variants_from_row(row, text)
    if not columns:
        return current

    updates = {
        "legal_terminology_rewrite": current.legal_terminology_rewrite,
        "regulatory_compliance_query": current.regulatory_compliance_query,
        "contract_clause_query": current.contract_clause_query,
        "risk_scenario_query": current.risk_scenario_query,
    }
    for column in columns:
        print(f"  Generating missing variant: {column}")
        generated = generate_variant_field(text, column, provider=provider)
        if not str(generated).strip():
            print(f"  [WARN] Empty response for {column}; keeping existing value.")
            continue
        updates[column] = clean_text(generated)
        time.sleep(SLEEP_SECONDS)

    return QueryVariants(**updates)


def collect_use_case_queries(use_case: str) -> list[dict]:
    uc = use_case.lower()
    level_configs = get_level_configs(uc)
    rows: list[dict] = []
    for level_name, cfg in level_configs.items():
        queries = load_queries_for_level(level_name, cfg["query_path"], cfg["gs_path"])
        for query in queries:
            rows.append({"level": level_name, "query": clean_text(query)})
    return rows


def ensure_query_variants(
    use_case: str,
    provider: str = "gemini",
    records_path: str | None = None,
    fill_missing_only: bool = True,
    force_regenerate: bool = False,
) -> Path:
    """Fill output_with_agents_<use_case>.csv with agent query variants.

    When fill_missing_only=True (default), only rows/fields that are empty or still
    equal to the original query are sent to the LLM. Progress is saved after each row.
    """
    path = Path(records_path) if records_path else recorded_queries_path_for(use_case)
    target_rows = collect_use_case_queries(use_case)

    if path.exists():
        records_df = pd.read_csv(path)
        records_df["level"] = records_df["level"].astype(str).str.lower()
        records_df["query"] = records_df["query"].apply(clean_text)
    else:
        records_df = pd.DataFrame(columns=["level", "query", *VARIANT_AGENT_COLUMNS])

    for column in VARIANT_AGENT_COLUMNS:
        if column not in records_df.columns:
            records_df[column] = ""

    updated_rows: list[dict] = []
    for target in target_rows:
        level = target["level"]
        query = target["query"]
        existing = records_df[
            (records_df["level"] == level) & (records_df["query"] == query)
        ]
        existing_row = existing.iloc[0].to_dict() if not existing.empty else None

        if force_regenerate:
            columns_to_generate = VARIANT_AGENT_COLUMNS
        elif fill_missing_only:
            columns_to_generate = missing_variant_columns(query, existing_row)
        else:
            columns_to_generate = VARIANT_AGENT_COLUMNS

        if columns_to_generate:
            print(f"[{use_case} / {level}] Generating variants for: {query[:90]}...")
            if columns_to_generate != VARIANT_AGENT_COLUMNS:
                print(f"  Missing columns: {', '.join(columns_to_generate)}")
            variants = generate_missing_variants(
                query,
                existing_row,
                provider=provider,
                columns=columns_to_generate,
            )
        else:
            variants = variants_from_row(existing_row, query)

        updated_rows.append(
            {
                "level": level,
                "query": query,
                "legal_terminology_rewrite": variants.legal_terminology_rewrite,
                "regulatory_compliance_query": variants.regulatory_compliance_query,
                "contract_clause_query": variants.contract_clause_query,
                "risk_scenario_query": variants.risk_scenario_query,
            }
        )

    pd.DataFrame(updated_rows).to_csv(path, index=False)
    print(f"Saved query variants to {path}")
    return path


def load_recorded_query_variants(use_case: str, records_path: str | None = None) -> dict[tuple[str, str], QueryVariants]:
    path = Path(records_path) if records_path else recorded_queries_path_for(use_case)
    if not path.exists():
        raise FileNotFoundError(
            f"Recorded query variants not found at {path}. "
            f"Run once with provider='ollama' or provider='gemini' to create it."
        )

    df = pd.read_csv(path)
    required_columns = {
        "level",
        "query",
        "legal_terminology_rewrite",
        "regulatory_compliance_query",
        "contract_clause_query",
        "risk_scenario_query",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Recorded query file {path} is missing required columns: {missing}")

    records: dict[tuple[str, str], QueryVariants] = {}
    for row in df.to_dict("records"):
        query = clean_text(row["query"])
        key = (str(row["level"]).strip().lower(), query)
        records[key] = QueryVariants(
            legal_terminology_rewrite=recorded_value(row["legal_terminology_rewrite"], query),
            regulatory_compliance_query=recorded_value(row["regulatory_compliance_query"], query),
            contract_clause_query=recorded_value(row["contract_clause_query"], query),
            risk_scenario_query=recorded_value(row["risk_scenario_query"], query),
        )
    print(f"Loaded recorded query variants from {path}")
    return records


def recorded_value(value, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    if str(value).strip() == "":
        return fallback
    return clean_text(value)


def query_variants_for(
    level_name: str,
    text: str,
    provider: str,
    recorded_variants: dict[tuple[str, str], QueryVariants] | None,
    recorded_rows: dict[tuple[str, str], dict] | None = None,
    fill_missing_only: bool = True,
) -> QueryVariants:
    if provider == "records":
        if recorded_variants is None:
            raise ValueError("recorded_variants must be loaded when provider='records'")

        key = (level_name.lower(), clean_text(text))
        if key not in recorded_variants:
            raise KeyError(
                f"No recorded query variants found for level='{level_name}' and query='{text[:120]}...'"
            )
        return recorded_variants[key]

    key = (level_name.lower(), clean_text(text))
    existing_row = recorded_rows.get(key) if recorded_rows else None
    if fill_missing_only:
        missing = missing_variant_columns(text, existing_row)
        if not missing:
            return variants_from_row(existing_row, text)
        return generate_missing_variants(text, existing_row, provider=provider, columns=missing)

    return generate_questions(text, provider=provider)


def bm25_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "level",
            "query",
            "rel_text",
            "score",
            "method",
            "query_variant",
            "source_variants",
        ],
    )


def concat_result_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return bm25_rows_to_df([])
    return pd.concat(frames, ignore_index=True)


def retrieved_passages_to_df(level_name: str, passages: list[RetrievedPassage]) -> pd.DataFrame:
    rows = [
        {
            "level": level_name,
            "query": passage.query,
            "rel_text": passage.rel_text,
            "score": passage.score,
            "method": passage.method,
            "query_variant": passage.query_variant,
            "source_variants": passage.query_variant,
        }
        for passage in passages
    ]
    return bm25_rows_to_df(rows)


def cross_encoder_rows_to_df(
    level_name: str,
    query_text: str,
    results: list[RankingResult],
    documents,
    cross_encoder,
    top_k: int,
    method: str,
    query_variant: str,
) -> pd.DataFrame:
    """Rerank BM25 candidates with the original query only."""
    unique_results: dict[int, RankingResult] = {}
    for result in results:
        unique_results.setdefault(result.document.doc_id, result)

    candidates = list(unique_results.values())
    if not candidates:
        return bm25_rows_to_df([])

    cross_inputs = [
        [query_text, documents[result.document.doc_id].text]
        for result in candidates
    ]
    cross_scores = cross_encoder.predict(cross_inputs)
    scored_results = [
        (result, float(score))
        for result, score in zip(candidates, cross_scores, strict=True)
    ]
    scored_results.sort(key=lambda item: item[1], reverse=True)

    rows = []
    for result, score in scored_results[:top_k]:
        rows.append(
            {
                "level": level_name,
                "query": query_text,
                "rel_text": documents[result.document.doc_id].text.replace("\n", " "),
                "score": score,
                "method": method,
                "query_variant": query_variant,
                "source_variants": result.document.doc_id,
            }
        )
    return bm25_rows_to_df(rows)


def result_sources(ranked_lists: list[RankedList], doc_id: int) -> str:
    return ", ".join(
        ranked_list.name
        for ranked_list in ranked_lists
        if any(result.document.doc_id == doc_id for result in ranked_list.results)
    )


def ranking_results_to_rows(
    level_name: str,
    query_text: str,
    results: list[RankingResult],
    documents,
    method: str,
    query_variant: str,
    source_variants,
) -> list[dict]:
    rows = []
    for item in results:
        sources = source_variants(item) if callable(source_variants) else source_variants
        rows.append(
            {
                "level": level_name,
                "query": query_text,
                "rel_text": documents[item.document.doc_id].text.replace("\n", " "),
                "score": item.score,
                "method": method,
                "query_variant": query_variant,
                "source_variants": sources,
            }
        )
    return rows


def to_eval_format(df: pd.DataFrame) -> pd.DataFrame:
    """Same columns as original algo_output *.xlsx for merge with gold standard."""
    out = df[["query", "rel_text", "score"]].copy()
    return out


def write_algo_output_files(
    use_case: str,
    output_frames: dict[str, pd.DataFrame],
) -> None:
    uc = use_case.lower()
    algo_dir = _algo_output_dir(uc)
    algo_dir.mkdir(parents=True, exist_ok=True)
    for level, suffix in LEVEL_TO_ALGO_SUFFIX.items():
        for method, df in output_frames.items():
            level_df = df[df["level"] == level] if not df.empty else df
            if level_df.empty:
                continue
            output_path = algo_dir / f"{uc}_{suffix}_algo_output_{method}.xlsx"
            to_eval_format(level_df).to_excel(output_path, index=False)
            print(f"Wrote {output_path}")


def load_queries_for_level(level_name: str, query_path: str | None, gs_path: str) -> list[str]:
    if query_path:
        df_queries = pd.read_excel(query_path)
        if "process_text" not in df_queries.columns:
            raise ValueError(f"Column 'process_text' not found in: {query_path}")
        queries = df_queries["process_text"].astype(str).tolist()
    else:
        df_gs = pd.read_excel(gs_path)
        if "query" not in df_gs.columns:
            raise ValueError(f"Column 'query' not found in: {gs_path}")
        queries = df_gs["query"].astype(str).drop_duplicates().tolist()

    queries = [clean_text(q) for q in queries]
    if NUM_ROWS is not None:
        queries = queries[:NUM_ROWS]
    print(f"Loaded {len(queries)} queries for level '{level_name}'")
    return queries


def run_level(
    level_name: str,
    queries: list[str],
    bm25_index,
    documents,
    sota_retriever: SotaRetriever,
    provider: str,
    top_k: int,
    recorded_variants: dict[tuple[str, str], QueryVariants] | None = None,
    recorded_rows: dict[tuple[str, str], dict] | None = None,
    fill_missing_only: bool = True,
    enable_llm_judge: bool = False,
    judge_provider: str | None = None,
    variants_save_path: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    rrf = ReciprocalRankFusion()
    all_bm25_baseline_rows: list[dict] = []
    all_bm25_rrf_rows: list[dict] = []
    all_bm25_rrf_weighted_rows: list[dict] = []
    bm25_ce_frames: list[pd.DataFrame] = []
    bm25_rrf_ce_frames: list[pd.DataFrame] = []
    bm25_rrf_weighted_ce_frames: list[pd.DataFrame] = []
    all_bm25_rrf_weighted_judged_rows: list[dict] = []
    bm25_rrf_weighted_ce_judged_frames: list[pd.DataFrame] = []
    query_rows: list[dict] = []

    for idx, text in enumerate(queries, 1):
        print(f"[{level_name}] Processing row {idx}/{len(queries)}...")
        agent_results = query_variants_for(
            level_name=level_name,
            text=text,
            provider=provider,
            recorded_variants=recorded_variants,
            recorded_rows=recorded_rows,
            fill_missing_only=fill_missing_only,
        )
        query_row = {
            "level": level_name,
            "query": text,
            "legal_terminology_rewrite": agent_results.legal_terminology_rewrite,
            "regulatory_compliance_query": agent_results.regulatory_compliance_query,
            "contract_clause_query": agent_results.contract_clause_query,
            "risk_scenario_query": agent_results.risk_scenario_query,
        }

        baseline_results = bm25_index.rank(Query(text=text), top_k=top_k)
        all_bm25_baseline_rows.extend(
            ranking_results_to_rows(
                level_name=level_name,
                query_text=text,
                results=baseline_results,
                documents=documents,
                method="bm25",
                query_variant="baseline",
                source_variants="baseline",
            )
        )

        bm25_ce_df = retrieved_passages_to_df(
            level_name,
            sota_retriever.search_bm25_ce(
                n=top_k,
                query=text,
                query_variant="baseline",
            ),
        )
        bm25_ce_frames.append(bm25_ce_df)

        query_lists = [
            ("legal_terminology_rewrite", agent_results.legal_terminology_rewrite),
            ("regulatory_compliance_query", agent_results.regulatory_compliance_query),
            ("contract_clause_query", agent_results.contract_clause_query),
            ("risk_scenario_query", agent_results.risk_scenario_query),
        ]
        ranked_lists: list[RankedList] = []
        for name, query_text in query_lists:
            ranked_lists.append(
                RankedList(
                    name=name,
                    results=bm25_index.rank(Query(text=clean_text(query_text)), top_k=top_k),
                )
            )

        merged_results = rrf.fuse(ranked_lists, top_k=top_k)
        merged_sources = lambda item: result_sources(ranked_lists, item.document.doc_id)
        bm25_rrf_ce_frames.append(
            cross_encoder_rows_to_df(
                level_name=level_name,
                query_text=text,
                results=merged_results,
                documents=documents,
                cross_encoder=sota_retriever.cross_encoder,
                top_k=top_k,
                method="bm25_rrf_ce",
                query_variant="query_merge_rrf",
            )
        )
        all_bm25_rrf_rows.extend(
            ranking_results_to_rows(
                level_name=level_name,
                query_text=text,
                results=merged_results,
                documents=documents,
                method="bm25_rrf",
                query_variant="query_merge_rrf",
                source_variants=merged_sources,
            )
        )

        ranked_lists_weighted: list[RankedList] = [
            RankedList(name="baseline", results=baseline_results),
            *ranked_lists,
        ]
        merged_weighted = rrf.fuse(
            ranked_lists_weighted,
            top_k=top_k,
            weights=WEIGHTED_RRF_WEIGHTS,
        )
        weighted_sources = lambda item: result_sources(ranked_lists_weighted, item.document.doc_id)
        bm25_rrf_weighted_ce_df = cross_encoder_rows_to_df(
            level_name=level_name,
            query_text=text,
            results=merged_weighted,
            documents=documents,
            cross_encoder=sota_retriever.cross_encoder,
            top_k=top_k,
            method="bm25_rrf_weighted_ce",
            query_variant="query_merge_rrf_weighted",
        )
        bm25_rrf_weighted_ce_frames.append(bm25_rrf_weighted_ce_df)
        all_bm25_rrf_weighted_rows.extend(
            ranking_results_to_rows(
                level_name=level_name,
                query_text=text,
                results=merged_weighted,
                documents=documents,
                method="bm25_rrf_weighted",
                query_variant="query_merge_rrf_weighted",
                source_variants=weighted_sources,
            )
        )

        if enable_llm_judge:
            if not judge_provider:
                raise ValueError("judge_provider is required when enable_llm_judge=True")

            judged_variants = llm_judge_refine_queries(
                JudgeContext(
                    original_query=text,
                    variants=agent_results,
                    bm25_results=baseline_results,
                    ce_rows=bm25_rrf_weighted_ce_df,
                ),
                documents=documents,
                judge_provider=judge_provider,
            )
            query_row.update(
                {
                    "judge_legal_terminology_rewrite": judged_variants.legal_terminology_rewrite,
                    "judge_regulatory_compliance_query": judged_variants.regulatory_compliance_query,
                    "judge_contract_clause_query": judged_variants.contract_clause_query,
                    "judge_risk_scenario_query": judged_variants.risk_scenario_query,
                }
            )

            judged_query_lists = [
                ("judge_legal_terminology_rewrite", judged_variants.legal_terminology_rewrite),
                ("judge_regulatory_compliance_query", judged_variants.regulatory_compliance_query),
                ("judge_contract_clause_query", judged_variants.contract_clause_query),
                ("judge_risk_scenario_query", judged_variants.risk_scenario_query),
            ]
            judged_ranked_lists = [
                RankedList(
                    name=name,
                    results=bm25_index.rank(Query(text=clean_text(query_text)), top_k=top_k),
                )
                for name, query_text in judged_query_lists
            ]
            judged_ranked_lists_weighted = [
                RankedList(name="baseline", results=baseline_results),
                *judged_ranked_lists,
            ]
            judged_merged_weighted = rrf.fuse(
                judged_ranked_lists_weighted,
                top_k=top_k,
                weights=WEIGHTED_RRF_WEIGHTS,
            )
            judged_sources = lambda item: result_sources(judged_ranked_lists_weighted, item.document.doc_id)
            bm25_rrf_weighted_ce_judged_frames.append(
                cross_encoder_rows_to_df(
                    level_name=level_name,
                    query_text=text,
                    results=judged_merged_weighted,
                    documents=documents,
                    cross_encoder=sota_retriever.cross_encoder,
                    top_k=top_k,
                    method="bm25_rrf_weighted_ce_judged",
                    query_variant="llm_judge_weighted_rrf",
                )
            )
            all_bm25_rrf_weighted_judged_rows.extend(
                ranking_results_to_rows(
                    level_name=level_name,
                    query_text=text,
                    results=judged_merged_weighted,
                    documents=documents,
                    method="bm25_rrf_weighted_judged",
                    query_variant="llm_judge_weighted_rrf",
                    source_variants=judged_sources,
                )
            )

        query_rows.append(query_row)

        if variants_save_path is not None:
            _upsert_variant_row(variants_save_path, query_row)

        if provider != "records":
            time.sleep(SLEEP_SECONDS)

    output_frames = {
        "bm25_baseline": bm25_rows_to_df(all_bm25_baseline_rows),
        "bm25_rrf": bm25_rows_to_df(all_bm25_rrf_rows),
        "bm25_rrf_weighted": bm25_rows_to_df(all_bm25_rrf_weighted_rows),
        "bm25_ce": concat_result_frames(bm25_ce_frames),
        "bm25_rrf_ce": concat_result_frames(bm25_rrf_ce_frames),
        "bm25_rrf_weighted_ce": concat_result_frames(bm25_rrf_weighted_ce_frames),
        "bm25_rrf_weighted_judged": bm25_rows_to_df(all_bm25_rrf_weighted_judged_rows),
        "bm25_rrf_weighted_ce_judged": concat_result_frames(bm25_rrf_weighted_ce_judged_frames),
    }
    return output_frames, query_rows


def _upsert_variant_row(path: Path, query_row: dict) -> None:
    if path.exists():
        records_df = pd.read_csv(path)
        records_df["level"] = records_df["level"].astype(str).str.lower()
        records_df["query"] = records_df["query"].apply(clean_text)
    else:
        records_df = pd.DataFrame(columns=["level", "query", *VARIANT_AGENT_COLUMNS])

    for column in VARIANT_AGENT_COLUMNS:
        if column not in records_df.columns:
            records_df[column] = ""

    level = str(query_row["level"]).lower()
    query = clean_text(query_row["query"])
    mask = (records_df["level"] == level) & (records_df["query"] == query)
    row_payload = {
        "level": level,
        "query": query,
        "legal_terminology_rewrite": query_row["legal_terminology_rewrite"],
        "regulatory_compliance_query": query_row["regulatory_compliance_query"],
        "contract_clause_query": query_row["contract_clause_query"],
        "risk_scenario_query": query_row["risk_scenario_query"],
    }

    if mask.any():
        for column, value in row_payload.items():
            records_df.loc[mask, column] = value
    else:
        records_df = pd.concat([records_df, pd.DataFrame([row_payload])], ignore_index=True)

    records_df.to_csv(path, index=False)


def load_recorded_rows_by_key(use_case: str, records_path: str | None = None) -> dict[tuple[str, str], dict]:
    path = Path(records_path) if records_path else recorded_queries_path_for(use_case)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    df["level"] = df["level"].astype(str).str.lower()
    df["query"] = df["query"].apply(clean_text)
    return {(str(row["level"]), clean_text(row["query"])): row for row in df.to_dict("records")}


def retrieval_pipeline(
    provider: str = LLM_PROVIDER,
    use_case: str = "uc1",
    records_path: str | None = None,
    enable_llm_judge: bool = False,
    judge_provider: str | None = None,
    fill_missing_variants_only: bool = True,
    skip_variant_generation: bool = False,
) -> None:
    provider = provider.lower()
    if judge_provider is not None:
        judge_provider = judge_provider.lower()
    if enable_llm_judge and judge_provider is None:
        judge_provider = provider if provider != "records" else LLM_PROVIDER

    uc = use_case.lower()
    corpus_path = corpus_path_for(uc)
    level_configs = get_level_configs(uc)
    variants_path = Path(records_path) if records_path else recorded_queries_path_for(uc)

    if skip_variant_generation:
        provider = "records"

    if provider == "records":
        recorded_variants = load_recorded_query_variants(uc, records_path=str(variants_path))
        recorded_rows = None
        variants_save_path = None
        variant_provider = "records"
    else:
        if fill_missing_variants_only:
            ensure_query_variants(
                use_case=uc,
                provider=provider,
                records_path=str(variants_path),
                fill_missing_only=True,
            )
            variant_provider = "records"
            recorded_variants = load_recorded_query_variants(uc, records_path=str(variants_path))
            recorded_rows = load_recorded_rows_by_key(uc, records_path=str(variants_path))
            variants_save_path = None
        else:
            recorded_variants = None
            recorded_rows = load_recorded_rows_by_key(uc, records_path=str(variants_path))
            variants_save_path = variants_path
            variant_provider = provider

    documents = load_corpus(corpus_path)
    bm25_index = build_bm25_index(corpus_path)
    sota_retriever = SotaRetriever([document.text for document in documents])
    all_query_rows: list[dict] = []
    frames_by_method: dict[str, list[pd.DataFrame]] = {method: [] for method in OUTPUT_SHEETS}

    for level_name, cfg in level_configs.items():
        queries = load_queries_for_level(level_name, cfg["query_path"], cfg["gs_path"])
        level_frames, level_query_rows = run_level(
            level_name=level_name,
            queries=queries,
            bm25_index=bm25_index,
            documents=documents,
            sota_retriever=sota_retriever,
            provider=variant_provider,
            top_k=cfg["top_k"],
            recorded_variants=recorded_variants,
            recorded_rows=recorded_rows,
            fill_missing_only=fill_missing_variants_only,
            enable_llm_judge=enable_llm_judge,
            judge_provider=judge_provider,
            variants_save_path=variants_save_path,
        )
        for method, df in level_frames.items():
            frames_by_method[method].append(df)
        all_query_rows.extend(level_query_rows)

    output_agents_path = variants_path
    output_ranking_path = f"output_ranking_{uc}.xlsx"
    if not skip_variant_generation and variants_save_path is not None:
        pd.DataFrame(all_query_rows).to_csv(output_agents_path, index=False)

    output_frames = {
        method: concat_result_frames(frames)
        for method, frames in frames_by_method.items()
    }

    with pd.ExcelWriter(output_ranking_path) as writer:
        for method, sheet_name in OUTPUT_SHEETS.items():
            output_frames[method].to_excel(writer, sheet_name=sheet_name, index=False)

    write_algo_output_files(uc, output_frames)

    print(
        f"Done. Saved: {output_agents_path}, {output_ranking_path}, "
        f"and algo_output under {_algo_output_dir(uc)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25/RRF + encoder retrieval pipeline for UC1/UC2/UC3.")
    parser.add_argument(
        "--use-case",
        choices=["uc1", "uc2", "uc3"],
        default="uc1",
        help="Which use case to run (default: uc1).",
    )
    parser.add_argument(
        "--provider",
        default=LLM_PROVIDER,
        help=f"Query variant provider: ollama, gemini, or records (default: {LLM_PROVIDER}).",
    )
    parser.add_argument(
        "--records-path",
        default=None,
        help="Optional CSV path for provider='records' (default: output_with_agents_<use-case>.csv).",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Run an LLM judge after the initial BM25+CE pass to refine the four query variants.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="LLM provider for --enable-llm-judge. Defaults to provider, or LLM_PROVIDER when provider='records'.",
    )
    args = parser.parse_args()
    retrieval_pipeline(
        provider=args.provider,
        use_case=args.use_case,
        records_path=args.records_path,
        enable_llm_judge=args.enable_llm_judge,
        judge_provider=args.judge_provider,
    )
