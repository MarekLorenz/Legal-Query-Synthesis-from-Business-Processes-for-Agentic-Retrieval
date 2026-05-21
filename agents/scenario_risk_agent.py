"""
Scenario / Risk-Based Agent
Expands the process into edge cases, risks, and violations
"""

from llm_provider import get_llm_provider


def build_prompt(process_text: str) -> str:
    return f"""
You are a legal risk and scenario analysis expert.

Your task is to expand the given business process into edge cases, risks, and legal violations for the sake of query expansion.

Requirements:
- Ask: What can go wrong? What legal consequences exist?
- Generate queries around: breaches, penalties, disputes, non-compliance
- Think about failure modes and their legal implications
- Frame in terms of consequences, remedies, and penalties
- Return ONLY the rewritten query, nothing else

Example transformation:
"late delivery" →
"legal consequences of delayed performance and breach of contract remedies"

Process/Query to expand into risks and scenarios:
\"\"\"{process_text}\"\"\"

Provide the risk and scenario-based perspective:
"""


def scenario_risk_agent(text: str, provider: str = "ollama") -> str:
    """Rewrites query focusing on risks, breaches, and consequences"""
    prompt = build_prompt(text)
    llm = get_llm_provider(provider)
    return llm.generate(prompt)
