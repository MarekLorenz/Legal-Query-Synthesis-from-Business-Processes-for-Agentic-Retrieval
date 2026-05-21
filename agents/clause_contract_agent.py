"""
Clause / Contract Perspective Agent
Rewrites the process as if searching for contract clauses
"""

from llm_provider import get_llm_provider


def build_prompt(process_text: str) -> str:
    return f"""
You are a legal query rewrite agent.

Your task is to rephrase the given query from a business process into a different but related query used for BM25 retrieval.

Frame the query as a relevant legal question that is optimized for retrieval from a regulatory corpus.

Make sure to stick closely to the intention of the business process description.

Process/Query to frame as contract clauses:
\"\"\"{process_text}\"\"\"

Provide the contract clause perspective:
"""


def clause_contract_agent(text: str, provider: str = "ollama") -> str:
    """Rewrites query as contract clause search"""
    prompt = build_prompt(text)
    llm = get_llm_provider(provider)
    return llm.generate(prompt)
