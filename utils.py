import spacy
import fitz  # PyMuPDF for PDF text extraction
import requests
import io
from difflib import SequenceMatcher
from typing import Dict, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def calculate_skill_similarity(skill1: str, skill2: str) -> float:
    """Calculate similarity between two skills using multiple methods"""
    # Normalize skills
    skill1 = skill1.lower().strip()
    skill2 = skill2.lower().strip()
    
    # Exact match
    if skill1 == skill2:
        return 1.0
    
    # Get word embeddings for semantic similarity
    doc1 = nlp(skill1)
    doc2 = nlp(skill2)
    
    # Calculate semantic similarity using word vectors
    semantic_sim = doc1.similarity(doc2) if doc1.vector_norm and doc2.vector_norm else 0
    
    # Calculate string similarity
    string_sim = SequenceMatcher(None, skill1, skill2).ratio()
    
    # Calculate TF-IDF similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([skill1, skill2])
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        tfidf_sim = 0
    
    # Combine similarities with weights
    combined_sim = (semantic_sim * 0.5 + string_sim * 0.3 + tfidf_sim * 0.2)
    
    # Ensure minimum similarity of 0.4 for related skills
    return max(0.4, combined_sim)

def analyze_skill_relationships(resume_skills: Set[str], job_requirements: Set[str]) -> float:
    """Analyze relationships between skills using semantic similarity"""
    if not resume_skills or not job_requirements:
        return 0.5  # Default to 50% if no skills to compare
    
    # Convert sets to lists for processing
    resume_list = list(resume_skills)
    job_list = list(job_requirements)
    
    # Calculate pairwise semantic similarities
    similarities = []
    for resume_skill in resume_list:
        resume_doc = nlp(resume_skill)
        for job_skill in job_list:
            job_doc = nlp(job_skill)
            if resume_doc.vector_norm and job_doc.vector_norm:
                sim = resume_doc.similarity(job_doc)
                similarities.append(sim)
    
    if not similarities:
        return 0.5
    
    # Calculate average similarity and adjust for skill coverage
    avg_similarity = sum(similarities) / len(similarities)
    skill_coverage = min(len(resume_skills) / len(job_requirements), 1.0)
    
    # Combine similarity and coverage
    return max(0.4, avg_similarity * 0.7 + skill_coverage * 0.3)

def compare_skills(resume_skills: Set[str], job_requirements: Set[str]) -> Tuple[Dict[str, str], float]:
    """Compare skills with a sophisticated scoring system"""
    if not job_requirements:
        return {}, 50.0  # Default to 50% if no requirements provided
    
    job_requirements = {skill.lower() for skill in job_requirements}
    resume_skills = {skill.lower() for skill in resume_skills}
    
    # Calculate individual skill matches
    skill_scores = {}
    total_score = 0
    
    for job_skill in job_requirements:
        best_match_score = 0
        best_match = None
        
        for resume_skill in resume_skills:
            similarity = calculate_skill_similarity(job_skill, resume_skill)
            if similarity > best_match_score:
                best_match_score = similarity
                best_match = resume_skill
        
        skill_scores[job_skill] = best_match_score
        total_score += best_match_score
    
    # Calculate base match percentage
    base_match = (total_score / len(job_requirements)) * 100
    
    # Analyze skill relationships
    relationship_score = analyze_skill_relationships(resume_skills, job_requirements)
    
    # Calculate final score with relationship adjustment
    adjusted_score = base_match * (0.7 + relationship_score * 0.3)
    
    # Ensure minimum score of 40%
    final_score = max(40.0, min(100.0, adjusted_score))
    
    # Create comparison dictionary with detailed feedback
    comparison = {
        skill: f"✔️ {score:.0%} Match" if score > 0.7 
        else f"⚠️ {score:.0%} Partial Match" if score > 0.4 
        else f"❌ {score:.0%} Low Match"
        for skill, score in skill_scores.items()
    }
    
    return comparison, round(final_score, 2)