from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

try:
    models = client.models.list()
    print("AVAILABLE MODELS:")
    for m in models:
        print("-", m.name)
except Exception as e:
    print("EXCEPTION:", e)
