import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


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
                    AI: """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input_message)

    try:
        # Ensure the response is a valid JSON list
        skills = json.loads(response.text)
        if isinstance(skills, list) and all(isinstance(skill, str) for skill in skills):
            return skills
        else:
            raise ValueError("Invalid format received from AI model")
    except json.JSONDecodeError:
        raise ValueError("Failed to parse AI response as JSON array")
