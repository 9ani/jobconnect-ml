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
    
    # Boost score for related skills
    if combined_sim > 0.1:  # Lower threshold for related skills
        # If skills are semantically related
        if semantic_sim > 0.2:
            combined_sim = max(combined_sim, 0.8)
        # If skills share common words or are part of the same domain
        elif string_sim > 0.2:
            combined_sim = max(combined_sim, 0.7)
    
    return combined_sim

def analyze_skill_relationships(resume_skills: Set[str], job_requirements: Set[str]) -> float:
    """Analyze relationships between skills using semantic similarity"""
    if not resume_skills or not job_requirements:
        return 0.0
    
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
        return 0.0
    
    # Calculate average similarity and adjust for skill coverage
    avg_similarity = sum(similarities) / len(similarities)
    skill_coverage = min(len(resume_skills) / len(job_requirements), 1.0)
    
    # Boost score for having related skills
    if skill_coverage > 0.1:  # Lower threshold
        avg_similarity = max(avg_similarity, 0.8)
    
    return avg_similarity * 0.5 + skill_coverage * 0.5

def compare_skills(resume_skills: Set[str], job_requirements: Set[str]) -> Tuple[Dict[str, str], float]:
    """Compare skills with a sophisticated scoring system"""
    if not job_requirements or not resume_skills:
        return {}, 0.0
    
    job_requirements = {skill.lower() for skill in job_requirements}
    resume_skills = {skill.lower() for skill in resume_skills}
    
    # Calculate individual skill matches
    skill_scores = {}
    total_score = 0
    matched_skills = 0
    core_skills = 0
    
    for job_skill in job_requirements:
        best_match_score = 0
        best_match = None
        
        for resume_skill in resume_skills:
            similarity = calculate_skill_similarity(job_skill, resume_skill)
            if similarity > best_match_score:
                best_match_score = similarity
                best_match = resume_skill
        
        skill_scores[job_skill] = best_match_score
        if best_match_score > 0.1:  # Lower threshold for matches
            matched_skills += 1
            if best_match_score == 1.0:  # Exact match
                core_skills += 1
        total_score += best_match_score
    
    # Calculate base match percentage with higher weight for core skills
    base_match = (total_score / len(job_requirements)) * 100
    
    # Analyze skill relationships
    relationship_score = analyze_skill_relationships(resume_skills, job_requirements)
    
    # Calculate final score with relationship adjustment and core skill boost
    core_skill_ratio = core_skills / len(job_requirements)
    adjusted_score = base_match * (0.5 + relationship_score * 0.3 + core_skill_ratio * 0.2)
    
    # Only return 0% if there are absolutely no matches
    if matched_skills == 0:
        final_score = 0.0
    else:
        # Boost score based on number of matched skills
        match_ratio = matched_skills / len(job_requirements)
        if match_ratio > 0.1:  # Lower threshold for boost
            adjusted_score *= 2.0  # Higher boost for related skills
        final_score = max(10.0, min(100.0, adjusted_score))
    
    # Create comparison dictionary with detailed feedback
    comparison = {
        skill: f"✔️ {score:.0%} Match" if score > 0.7 
        else f"⚠️ {score:.0%} Partial Match" if score > 0.4 
        else f"❌ {score:.0%} Low Match"
        for skill, score in skill_scores.items()
    }
    
    return comparison, round(final_score, 2)