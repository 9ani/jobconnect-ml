from fastapi import FastAPI, HTTPException
from models import ResumeRequest, JobDescriptionRequest
from utils import extract_text_from_pdf_url, extract_skills, compare_skills
from gemini_service import extract_skills_from_description

app = FastAPI(title="Resume Skill Matching API")

@app.post("/match-skills/")
def match_resume_skills(request: ResumeRequest):
    try:
        resume_text = extract_text_from_pdf_url(request.pdf_url)
        resume_skills = extract_skills(resume_text)
        comparison, match_percentage = compare_skills(resume_skills, request.job_requirements)
        return {"extracted_skills": list(resume_skills), "comparison_result": comparison, "Percentage": match_percentage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-job-skills/")
def extract_job_skills(request: JobDescriptionRequest):
    try:
        skills = extract_skills_from_description(request.job_description)
        return skills  # Directly return the list instead of wrapping it in a dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
