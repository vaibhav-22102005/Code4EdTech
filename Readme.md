Auto Resume Relevance Checker
This is an artificial intelligence engine specially created for the purpose of auto-screening of resumes. It estimates a relevance score between a candidate resume and a chosen job description, makes the decision of suitability, and returns personalized comments back to the candidate.

1. Problem Statement

Recruitment and placement groups usually receive a large number of applications for every job posting. It is manually time-consuming and inconsistent, as well as not scalable, to check each resume against a job description (JD). Automating the process alleviates delays in shortlisting candidates and overloading of personnel with work, diverting their time away from value-added activity such as counseling students and preparing for interviews. The objective of this project is to do away with that with a system that is consistent, scalable, and automated in evaluation.

2. Our Approach

The software uses a hybrid method to score and process resumes with a balance between specificity and contextual understanding. The entire work flow is taken as a stateful graph using LangGraph, with the analysis proceeding through a series of clear, logical stages:

JD Parsing: We start with applying a Large Language Model (LLM) to conscientiously select the top, "must-have" skills from the job description.

Hard Match Analysis: It conducts a keyword search to determine how many of these skills extracted are actually named in the resume. This gives a base score based on hard evidence.

Semantic Match Analysis: To get an idea about the context other than keywords, the system uses embedding models to convert the resume and JD into vectors. It then calculates the semantic similarity between them and scores the resume according to the extent of the match between the candidate's experience and the job irrespective of the usage of keywords.

Weighted Scoring & Verdict: We sum the hard match score and the semantic score with a weighted algorithm (hard match: 40%, semantic match: 60%) and determine an end relevance score. We assign a verdict of "High," "Medium," or "Low Suitability" based upon the score.

Feedback Generation: Ultimately, the LLM produces customized, positive feedback for the learner, indicating omitted skills and potential enhancement zones.

3. Tech Stack, APIs, and Services

This project uses a contemporary stack of web development and AI tools to provide a turnkey solution.

Programming Language: Python
================

AI Orchestration

LangChain: The core package for building and manipulating interactions with the language models.

LangGraph: Was applied for modeling the multi-step analysis process as a robust, stateful workflow.

Language APIs & Models:

Google Gemini 1.5 Flash: The LLM that was applied for all reasoning work, including parsing the job description, computing the semantic fit, and generating the student comments.

Hugging Face Embeddings (sentence-transformers/all-MiniLM-L6-v2): An open-source and fast model that is applied in order to create the vector embeddings for the semantic similarity test.

Web Framework:

Streamlit: A Python library used to rapidly build and deploy the interactive web user interface.

Document Analysis:

PyPDF2 & python-docx: These libraries are used to get the raw text from uploaded DOCX and PDF resume files.

Development & Debugging:

LangSmith: (Optional but Recommended) An observability system for debugging and tracing the graphs and chains that are constructed with LangChain.



5. How to Use

Uploading of Job Description: You may upload a .txt file of the job description or just type and copy and paste it into the text box on the left.

Upload Resume: Upload the resume of the candidate on the right. The application supports both .pdf and .docx file types.

Analyze: Go into the Resume Wizard and click on the "Analyze Resume" button. View Results: The software will indicate the final relevance score, a suitability decision, and an extensive breakdown including the keyword match score, semantic fit score, list of missing skills, and tailored comments.