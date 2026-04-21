from google import genai
from dotenv import load_dotenv

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
)
print(response.text)