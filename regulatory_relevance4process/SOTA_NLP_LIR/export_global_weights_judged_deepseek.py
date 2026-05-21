"""Export DeepSeek/Ollama global-weights judged pipeline artifacts for query diversification."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOTA_DIR = Path(__file__).resolve().parent
DIVERSIFICATION_SUBDIR = "global_weights_judged_deepseek"
JUDGE_COLUMN_PREFIX = "ollama_global_judge"
GLOBAL_RANKING_SHEET = "Global_linear_RRF_judged"

AGENT_VARIANT_COLUMNS = [
    "legal_terminology_rewrite",
    "regulatory_compliance_query",
    "contract_clause_query",
    "risk_scenario_query",
]


def find_project_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    while root.name != "Legal-Query-Synthesis-from-Business-Processes-for-Agentic-Retrieval" and root.parent != root:
        root = root.parent
    if root.name != "Legal-Query-Synthesis-from-Business-Processes-for-Agentic-Retrieval":
        raise RuntimeError("Could not find Legal-Query-Synthesis-from-Business-Processes-for-Agentic-Retrieval")
    return root


def diversification_dir(project_root: Path) -> Path:
    path = (
        project_root
        / "regulatory_relevance4process/SOTA_NLP_LIR/query_diversification_results"
        / DIVERSIFICATION_SUBDIR
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def variant_text(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return str(row["query"]).strip()
    value = row[column]
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return str(row["query"]).strip()
    return str(value).strip()


def judged_variants_export_df(records_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in records_df.iterrows():
        export_row = {"level": row["level"], "query": row["query"]}
        for field in AGENT_VARIANT_COLUMNS:
            judged_col = f"{JUDGE_COLUMN_PREFIX}_{field}"
            if judged_col in row.index and pd.notna(row[judged_col]) and str(row[judged_col]).strip():
                export_row[field] = row[judged_col]
            else:
                export_row[field] = variant_text(row, field)
        rows.append(export_row)
    return pd.DataFrame(rows)


def predictions_to_diversification_ranking_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "level": pred_df["level"],
            "query": pred_df["query"],
            "rel_text": pred_df["rel_text"],
            "score": pred_df["score"],
            "method": "global_linear_rrf_judged_deepseek",
            "query_variant": "global_linear_rrf_judged_deepseek",
            "source_variants": "ollama_global_judge_variants",
            "rank": pred_df["rank"],
            "split": pred_df["split"],
        }
    )


def export_global_weights_judged_deepseek(
    records_by_use_case: dict[str, pd.DataFrame],
    predictions_df: pd.DataFrame,
    project_root: Path,
) -> list[Path]:
    out_dir = diversification_dir(project_root)
    written: list[Path] = []

    for use_case in sorted(records_by_use_case.keys()):
        records_df = records_by_use_case[use_case]

        agents_path = out_dir / f"output_with_agents_{use_case}_global_judge_ranking.csv"
        judged_variants_export_df(records_df).to_csv(agents_path, index=False)
        written.append(agents_path)
        print(f"Exported {agents_path} — {len(records_df)} rows (judged variants)")

        uc_predictions = predictions_df[predictions_df["use_case"] == use_case].copy()
        ranking_path = out_dir / f"output_ranking_{use_case}_global_judge.xlsx"
        ranking_df = predictions_to_diversification_ranking_df(uc_predictions)
        with pd.ExcelWriter(ranking_path) as writer:
            ranking_df.to_excel(writer, sheet_name=GLOBAL_RANKING_SHEET, index=False)
        written.append(ranking_path)
        print(
            f"Exported {ranking_path} — {len(ranking_df)} rows "
            f"(sheet '{GLOBAL_RANKING_SHEET}', column 'score')"
        )

    return written


def load_records_by_use_case(project_root: Path, use_cases: list[str]) -> dict[str, pd.DataFrame]:
    records_by_use_case: dict[str, pd.DataFrame] = {}
    for use_case in use_cases:
        path = project_root / f"output_with_agents_{use_case}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        records_by_use_case[use_case] = pd.read_csv(path)
    return records_by_use_case


def load_predictions_from_results(project_root: Path, judge_provider: str = "ollama") -> pd.DataFrame:
    results_path = project_root / f"linear_rrf_global_weights_{judge_provider}_judge_results.xlsx"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")
    return pd.read_excel(results_path, sheet_name="ranked_predictions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export global_weights_judged_deepseek diversification files.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repo root (default: auto-detect from cwd).",
    )
    parser.add_argument(
        "--results-xlsx",
        type=Path,
        default=None,
        help="Workbook with sheet ranked_predictions (default: linear_rrf_global_weights_ollama_judge_results.xlsx).",
    )
    parser.add_argument("--use-cases", nargs="+", default=["uc1", "uc2", "uc3"])
    args = parser.parse_args()

    project_root = args.project_root or find_project_root()
    records_by_use_case = load_records_by_use_case(project_root, args.use_cases)

    if args.results_xlsx:
        predictions_df = pd.read_excel(args.results_xlsx, sheet_name="ranked_predictions")
    else:
        predictions_df = load_predictions_from_results(project_root)

    export_global_weights_judged_deepseek(records_by_use_case, predictions_df, project_root)


if __name__ == "__main__":
    main()
