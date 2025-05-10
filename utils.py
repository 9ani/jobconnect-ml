import spacy
import fitz  # PyMuPDF for PDF text extraction
import requests
import io
from difflib import SequenceMatcher

# Load the custom-trained NLP model
nlp = spacy.load("en_resume_ner")

# Function to download and extract text from a PDF URL
def extract_text_from_pdf_url(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_file = io.BytesIO(response.content)  # Load file into memory
        doc = fitz.open(stream=pdf_file, filetype="pdf")  # Open PDF from memory
        text = "\n".join(page.get_text("text") for page in doc)  # Extract text from pages
        return text.strip()
    else:
        raise Exception(f"Failed to download PDF. Status Code: {response.status_code}")

# Function to extract skills from resume text
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text.strip().lower() for ent in doc.ents if ent.label_ == "SKILLS"]
    return set(" ".join(skill.split()) for skill in skills)  # Normalize spacing



def is_strong_match(skill1, skill2, threshold=0.75):
    """Check if two skills have a strong similarity score"""
    ratio = SequenceMatcher(None, skill1.lower(), skill2.lower()).ratio()
    return ratio >= threshold

def compare_skills(resume_skills, job_requirements):
    job_requirements = set(skill.lower() for skill in job_requirements)  # Normalize case
    matched_skills = set()

    for job_skill in job_requirements:
        job_words = set(job_skill.lower().split())  # Break into words
        for resume_skill in resume_skills:
            resume_words = set(resume_skill.lower().split())

            # Check if at least one word matches exactly
            if job_words & resume_words:
                matched_skills.add(job_skill)
                break

            # Check for strong similarity
            if any(is_strong_match(word1, word2) for word1 in job_words for word2 in resume_words):
                matched_skills.add(job_skill)
                break

    comparison = {skill: "✔️ Match" if skill in matched_skills else "❌ No Match" for skill in job_requirements}
    match_percentage = round((len(matched_skills) / len(job_requirements)) * 100, 2) if job_requirements else 0

    return comparison, match_percentage