"""
Regulatory/Compliance Agent
Rewrites the query as a compliance requirement search
"""


def build_prompt(process_text: str) -> str:
    return f"""
You are a regulatory and compliance expert.

Your task is to rewrite the given business process as a compliance requirement search query.

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


def regulatory_compliance_agent(client, text: str) -> str:
    """Rewrites query as compliance requirement search"""
    prompt = build_prompt(text)
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Regulatory Compliance Agent failed: {e}")
        return ""
