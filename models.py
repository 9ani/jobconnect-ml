from pydantic import BaseModel
from typing import List

class ResumeRequest(BaseModel):
    pdf_url: str
    job_requirements: List[str]

class JobDescriptionRequest(BaseModel):
    job_description: str
