"""
Clause / Contract Perspective Agent
Rewrites the process as if searching for contract clauses
"""


def build_prompt(process_text: str) -> str:
    return f"""
You are a contracts and legal document expert.

Your task is to rewrite the given business process as a contract clause search query.

Requirements:
- Convert steps into: obligations, rights, liabilities
- Add clause-like phrasing (e.g., "clause", "provision", "terms")
- Frame in terms of contractual relationships and responsibilities
- Think about what clauses would govern this behavior in agreements
- Return ONLY the rewritten query, nothing else

Example transformation:
"refund customer" →
"refund obligations and liability clauses in service agreements"

Process/Query to frame as contract clauses:
\"\"\"{process_text}\"\"\"

Provide the contract clause perspective:
"""


def clause_contract_agent(client, text: str) -> str:
    """Rewrites query as contract clause search"""
    prompt = build_prompt(text)
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Clause/Contract Agent failed: {e}")
        return ""
