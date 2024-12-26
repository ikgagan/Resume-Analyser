import os
import re
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import openai
from dotenv import load_dotenv  # Import the dotenv library

# Load environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API key not found. Please check your .env file.")
openai.api_key = OPENAI_API_KEY

def parse_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def fetch_job_description(url):
    """Fetch job description content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text content from the webpages
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching job description: {e}")
        return None

def analyze_fit(resume_text, job_description):
    """Analyze the fit between resume and job description using OpenAI GPT."""
    prompt = (
        "Analyze the fit between the following resume and job description. "
        "Provide a score out of 100 and detailed feedback.\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_description}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error analyzing fit: {e}")
        return None

def generate_cover_letter(resume_text, job_description):
    """Generate a cover letter for the given resume and job description."""
    prompt = (
        "Using the following resume and job description, generate a professional cover letter tailored to the job:\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_description}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating cover letter: {e}")
        return None

def main():
    print("Resume Analyser Tool")
    print("---------------------")

    # Input: Resume file
    resume_path = input("Enter the path to the resume PDF: ")
    if not os.path.exists(resume_path):
        print("Error: Resume file not found.")
        return
    
    resume_text = parse_pdf(resume_path)

    # Input: Job description URL
    job_url = input("Enter the job description URL: ")
    job_description = fetch_job_description(job_url)
    if not job_description:
        print("Error: Failed to fetch job description.")
        return

    # Analyze fit
    print("Analyzing fit...")
    fit_analysis = analyze_fit(resume_text, job_description)
    if fit_analysis:
        print("\nFit Analysis:")
        print(fit_analysis)

    # Generate cover letter
    generate_letter = input("\nWould you like to generate a cover letter? (yes/no): ").strip().lower()
    if generate_letter == "yes":
        print("Generating cover letter...")
        cover_letter = generate_cover_letter(resume_text, job_description)
        if cover_letter:
            print("\nCover Letter:")
            print(cover_letter)

if __name__ == "__main__":
    main()
