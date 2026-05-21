"""
Legal Terminology Rewriter Agent (Domain Translator)
Translates business/process language → formal legal language
"""

from llm_provider import get_llm_provider


def build_prompt(process_text: str) -> str:
    return f"""
You are a query diversification expert. Your task is to rephrase the given query from a business process into a different but related query used for BM25 retrieval.

The domain is legal information retrieval and the corpus contains regulatory text passages, i.e. articles, laws etc.

Re-express the input query using legal terminology and regulatory language while preserving its original intent. The rewritten query should better match how concepts appear in legal and regulatory documents.

Requirements:
- Map business terms to legal equivalents
- Introduce legal phrasing and formal jargon
- Maintain the core meaning while using legal vocabulary
- Prevent topic drift
- Return ONLY the rewritten query, nothing else

Example transformations:
- "cancel order" → "contract termination"
- "user data handling" → "processing of personal data"

Process/Query to translate:
\"\"\"{process_text}\"\"\"

Provide the legal terminology rewrite:
"""


def legal_terminology_rewriter(text: str, provider: str = "ollama") -> str:
    """Rewrites query using legal terminology"""
    prompt = build_prompt(text)
    llm = get_llm_provider(provider)
    return llm.generate(prompt)
