import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Listing available models...")
try:
    with open("models.txt", "w") as f:
        for m in genai.list_models():
            f.write(f"Model: {m.name}\n")
            f.write(f"Methods: {m.supported_generation_methods}\n")
            f.write("-" * 20 + "\n")
    print("Models written to models.txt")
except Exception as e:
    print(f"Error: {e}")
