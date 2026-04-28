"""
Scenario / Risk-Based Agent
Expands the process into edge cases, risks, and violations
"""


def build_prompt(process_text: str) -> str:
    return f"""
You are a legal risk and scenario analysis expert.

Your task is to expand the given business process into edge cases, risks, and legal violations.

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


def scenario_risk_agent(client, text: str) -> str:
    """Rewrites query focusing on risks, breaches, and consequences"""
    prompt = build_prompt(text)
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Scenario/Risk Agent failed: {e}")
        return ""
