---
title: Jobconnect
emoji: 💻
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# JobConnect-ML

JobConnect-ML is a machine learning-powered backend service that extracts and matches skills from resumes using NLP. It is built with FastAPI and integrates with AI models for skill extraction.

## Features
- Extracts skills from resumes using NLP (Spacy and Hugging Face models)
- Matches extracted skills with job descriptions
- Provides a match percentage between resumes and job requirements
- Uses Google Generative AI for advanced text processing

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/9ani/jobconnect-ml.git
cd jobconnect-ml
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download and Install Hugging Face Model
```bash
pip install https://huggingface.co/JeswinMS4/en_resume_ner/resolve/main/en_resume_ner-any-py3-none-any.whl
```

### 5. Set Up Environment Variables
Create a `.env` file in the project root and add the required API keys:
```
GOOGLE_API_KEY=your_google_api_key
```

## Running the Application
Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```

## API Endpoints

### 1. Match Skills with Job Description
**Endpoint:** `POST /match-skills/`  
**Request Body:**
```json
{
  "pdf_url": "https://example.com/resume.pdf",
  "job_requirements": "Software Engineer with Python experience"
}
```
**Response:**
```json
{
  "extracted_skills": ["Python", "Machine Learning", "FastAPI"],
  "comparison_result": "Matching skills found",
  "matchPercentage": 85
}
```

## Contributing
Feel free to submit issues and pull requests to improve the project.

## License
This project is licensed under the MIT License.
