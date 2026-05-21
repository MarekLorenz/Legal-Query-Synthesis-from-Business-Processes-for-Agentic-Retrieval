"""
Regulatory/Compliance Agent
Rewrites the query as a compliance requirement search
"""

from llm_provider import get_llm_provider


def build_prompt(process_text: str) -> str:
    return f"""
You are a regulatory and compliance expert.

Your task is to rewrite the given business process as a compliance requirement search query for the sake of query expansion.

Requirements:
- Frame the process in terms of: obligations, restrictions, reporting requirements
- Focus on regulatory compliance perspective
- Use regulatory language like "legal requirements", "must", "required", "obligations"
- Think about GDPR, CCPA, industry regulations
- Return ONLY the rewritten query, nothing else

Example transformation:
"store customer data" →
"legal requirements for storage and retention of personal data under GDPR"

Process/Query to reframe:
\"\"\"{process_text}\"\"\"

Provide the compliance/regulatory requirements framing:
"""


def regulatory_compliance_agent(text: str, provider: str = "ollama") -> str:
    """Rewrites query as compliance requirement search"""
    prompt = build_prompt(text)
    llm = get_llm_provider(provider)
    return llm.generate(prompt)
