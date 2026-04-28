"""
Legal Terminology Rewriter Agent (Domain Translator)
Translates business/process language → formal legal language
"""


def build_prompt(process_text: str) -> str:
    return f"""
You are a legal language expert specializing in business process documentation.

Your task is to translate the given business/process language into formal legal terminology.

Requirements:
- Map business terms to legal equivalents
- Introduce legal phrasing and formal jargon
- Maintain the core meaning while using legal vocabulary
- Return ONLY the rewritten query, nothing else

Example transformations:
- "cancel order" → "contract termination"
- "user data handling" → "processing of personal data"

Process/Query to translate:
\"\"\"{process_text}\"\"\"

Provide the legal terminology rewrite:
"""


def legal_terminology_rewriter(client, text: str) -> str:
    """Rewrites query using legal terminology"""
    prompt = build_prompt(text)
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Legal Terminology Rewriter failed: {e}")
        return ""
