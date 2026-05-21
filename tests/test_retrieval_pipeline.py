import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_main():
    existing = sys.modules.get("main")
    if existing is not None and not hasattr(existing, "missing_variant_columns"):
        del sys.modules["main"]
    import main

    if not hasattr(main, "missing_variant_columns"):
        return importlib.reload(main)
    return main


main = _load_main()
from agents.query_judge_agent import JudgeContext, query_judge_agent  # noqa: E402


def test_clean_text_collapses_newlines():
    raw = "store data\n\nand\nprocess\nrequests"
    assert main.clean_text(raw) == "store data and process requests"


def test_missing_variant_columns_flags_unset_and_duplicate_values():
    query = "collect customer consent"
    row = {
        "legal_terminology_rewrite": query,
        "regulatory_compliance_query": "GDPR consent obligations",
        "contract_clause_query": None,
        "risk_scenario_query": "",
    }
    missing = main.missing_variant_columns(query, row)
    assert "legal_terminology_rewrite" in missing
    assert "contract_clause_query" in missing
    assert "risk_scenario_query" in missing
    assert "regulatory_compliance_query" not in missing


@patch("agents.query_judge_agent.get_llm_provider")
def test_query_judge_agent_parses_llm_json(mock_get_provider):
    @dataclass
    class StubVariants:
        legal_terminology_rewrite: str = "legal original"
        regulatory_compliance_query: str = "regulatory original"
        contract_clause_query: str = "contract original"
        risk_scenario_query: str = "risk original"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = """
    {
      "legal_terminology_rewrite": "refined legal",
      "regulatory_compliance_query": "refined regulatory",
      "contract_clause_query": "refined contract",
      "risk_scenario_query": "refined risk"
    }
    """
    mock_get_provider.return_value = mock_llm

    context = JudgeContext(
        original_query="store customer data",
        variants=StubVariants(),
        bm25_results=[],
        ce_rows=pd.DataFrame(),
    )
    refined = query_judge_agent(context, documents={}, provider="ollama")

    assert refined.legal_terminology_rewrite == "refined legal"
    assert refined.regulatory_compliance_query == "refined regulatory"
    assert refined.contract_clause_query == "refined contract"
    assert refined.risk_scenario_query == "refined risk"
    mock_get_provider.assert_called_once_with("ollama")
