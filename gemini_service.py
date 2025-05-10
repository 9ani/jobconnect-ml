import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# print("Available models:")
# for model in genai.list_models():
#     print(f"- {model.name} (supports: {model.supported_generation_methods})")
#
#
def extract_skills_from_description(summary: str):
    input_message = f"""USER: Imagine you're an NER AI model. 
Your task is to extract technical skills, frameworks, languages, software, and concepts found in the given job posting. 
You are allowed to change the names of skills and software to be standard and meaningful.
Make a single JSON array.
The goal is so that the users will get an overview of the skills they need to have.
Do not write sentences, only 1-3 word entities.
Format your response strictly as a JSON array of strings.

Example response:
["Python", "FastAPI", "PostgreSQL", "AWS", "Docker", "CI/CD"]

USER: Here is the posting: ```{summary}```
AI:"""

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    try:
        response = model.generate_content(input_message)

        raw = response.text.strip()
        print("Gemini raw response:", raw)

        # 🧼 Clean up: Remove markdown fences like ```json ... ```
        if raw.startswith("```"):
            raw = raw.strip("```json").strip("```")

        # 🧠 Try to find first valid JSON array using regex (more forgiving)
        import re
        match = re.search(r'\[(.*?)\]', raw, re.DOTALL)
        if match:
            cleaned_json = f"[{match.group(1)}]"
        else:
            raise ValueError("Response does not contain a valid JSON array")

        skills = json.loads(cleaned_json)

        if isinstance(skills, list) and all(isinstance(skill, str) for skill in skills):
            return skills
        else:
            raise ValueError("AI response is not a valid list of strings")

    except json.JSONDecodeError as e:
        print("JSON parse error:", str(e))
        raise ValueError("Failed to parse AI response as JSON array")

    except Exception as e:
        print("Gemini content generation error:", str(e))
        raise e
