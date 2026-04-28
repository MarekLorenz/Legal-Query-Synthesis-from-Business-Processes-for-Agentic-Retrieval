import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

from agents.legal_terminology_rewriter import legal_terminology_rewriter
from agents.regulatory_compliance_agent import regulatory_compliance_agent
from agents.clause_contract_agent import clause_contract_agent
from agents.scenario_risk_agent import scenario_risk_agent


# ----------------------------
# Config
# ----------------------------
INPUT_PATH = "data/Input_queries_low_uc1.xlsx"
OUTPUT_PATH = "data/output_with_agents.csv"
MODEL_NAME = "gemini-3-flash-preview"
NUM_ROWS = 1
SLEEP_SECONDS = 1


# ----------------------------
# Setup
# ----------------------------
def init_client():
    load_dotenv()
    return genai.Client()


# ----------------------------
# Prompt + Generation
# ----------------------------
def generate_questions(client, text: str) -> dict:
    """Call all 4 agents to generate different query perspectives"""
    
    # Call all 4 agents
    legal_term = legal_terminology_rewriter(client, text)
    regulatory = regulatory_compliance_agent(client, text)
    contract_clause = clause_contract_agent(client, text)
    risk_scenario = scenario_risk_agent(client, text)
    
    return {
        "legal_terminology_rewrite": legal_term,
        "regulatory_compliance_query": regulatory,
        "contract_clause_query": contract_clause,
        "risk_scenario_query": risk_scenario
    }


# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    # Load data
    df = pd.read_excel(INPUT_PATH)

    if "process_text" not in df.columns:
        raise ValueError("Column 'process_text' not found")

    df = df.head(NUM_ROWS).copy()

    client = init_client()

    results = {
        "legal_terminology_rewrite": [],
        "regulatory_compliance_query": [],
        "contract_clause_query": [],
        "risk_scenario_query": []
    }

    for idx, text in enumerate(df["process_text"], 1):
        print(f"Processing row {idx}/{len(df)}...")
        agent_results = generate_questions(client, text)

        results["legal_terminology_rewrite"].append(agent_results["legal_terminology_rewrite"])
        results["regulatory_compliance_query"].append(agent_results["regulatory_compliance_query"])
        results["contract_clause_query"].append(agent_results["contract_clause_query"])
        results["risk_scenario_query"].append(agent_results["risk_scenario_query"])

        time.sleep(SLEEP_SECONDS)  # rate limiting

    # Attach results
    for key, values in results.items():
        df[key] = values

    # Save as CSV
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Done. Output saved to: {OUTPUT_PATH}")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()