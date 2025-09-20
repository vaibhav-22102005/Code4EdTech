import os
import re
import io
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from typing import TypedDict, List
from dotenv import load_dotenv


import PyPDF2
import docx


from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings


load_dotenv()


google_api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

if not google_api_key:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

try:

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

except Exception as e:
    st.error(f"Error initializing models. Is your Google API Key correct? Error: {e}")
    st.stop()

def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def extract_text_from_docx(file_bytes):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return None



class AnalysisState(TypedDict):
    """Defines the state for our LangGraph agent."""
    resume_text: str
    jd_text: str
    extracted_skills: List[str]
    hard_match_score: float
    semantic_score: float
    final_score: int
    verdict: str
    missing_elements: str
    feedback: str
    error: str



def parse_jd_node(state: AnalysisState):
    """Extracts key skills and qualifications from the Job Description using an LLM."""
    prompt = ChatPromptTemplate.from_template(
        "You are an expert recruitment assistant. Extract the most important 'must-have' skills and qualifications from this job description. "
        "List them as a comma-separated string. For example: 'Python, SQL, Data Analysis, Machine Learning, AWS'.\n\n"
        "Job Description:\n{jd}"
    )
    chain = prompt | llm
    try:
        result = chain.invoke({"jd": state["jd_text"]})
        skills = [skill.strip() for skill in result.content.split(',')]
        return {"extracted_skills": skills}
    except Exception as e:
        return {"error": f"Failed to parse JD: {e}"}


def hard_match_node(state: AnalysisState):
    """Performs a keyword-based 'hard match' for skills."""
    if state.get("error"): return {} 
    
    resume_lower = state["resume_text"].lower()
    found_skills = 0
    total_skills = len(state["extracted_skills"])
    
    if total_skills == 0:
        return {"hard_match_score": 0.0}

    for skill in state["extracted_skills"]:
       
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_lower):
            found_skills += 1
            
    score = (found_skills / total_skills) * 100 if total_skills > 0 else 0
    return {"hard_match_score": score}

def semantic_match_node(state: AnalysisState):
    """Performs a semantic similarity check using embeddings."""
    if state.get("error"): return {} 
    
    try:
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.create_documents([state["resume_text"]])
        vector_store = FAISS.from_documents(documents, embeddings_model)
        retriever = vector_store.as_retriever(k=5) 

        
        prompt = ChatPromptTemplate.from_template(
            "Based on the following resume context, how well does the candidate's experience align with the job description? "
            "Answer with a percentage score from 0 to 100, where 100 is a perfect match. "
            "Respond with ONLY the number (e.g., '85').\n\n"
            "**Resume Context:**\n{context}\n\n"
            "**Job Description:**\n{jd}"
        )
        doc_chain = create_stuff_documents_chain(llm, prompt)
        
        
        relevant_docs = retriever.invoke(state["jd_text"])
        
        
        result = doc_chain.invoke({
            "context": relevant_docs,
            "jd": state["jd_text"]
        })

        
        score_match = re.search(r'\d+', result)
        score = float(score_match.group(0)) if score_match else 0.0
        
        return {"semantic_score": score}
    except Exception as e:
        return {"error": f"Failed during semantic match: {e}"}

def scoring_node(state: AnalysisState):
    """Calculates the final weighted score and verdict."""
    if state.get("error"): return {}
    
    
    hard_score = state.get("hard_match_score", 0.0)
    semantic_score = state.get("semantic_score", 0.0)
    
    final_score = int((hard_score * 0.4) + (semantic_score * 0.6))
    
    verdict = "Low Suitability"
    if final_score >= 75:
        verdict = "High Suitability"
    elif 50 <= final_score < 75:
        verdict = "Medium Suitability"
        
    return {"final_score": final_score, "verdict": verdict}

def feedback_node(state: AnalysisState):
    """Generates personalized feedback for the student."""
    if state.get("error"): return {}
    
    
    resume_lower = state["resume_text"].lower()
    missing_skills = [
        skill for skill in state["extracted_skills"] 
        if not re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_lower)
    ]
    missing_elements_str = ", ".join(missing_skills) if missing_skills else "None identified."
    
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful career coach. The candidate's resume has been evaluated against a job description. "
        "Their final score is {score}/100 ({verdict}).\n"
        "The following required skills seem to be missing or not clearly stated: {missing_elements}\n\n"
        "Based on this, provide 2-3 concise, actionable suggestions for the student to improve their resume or skills for this type of role. "
        "Keep the tone encouraging and constructive."
    )
    chain = prompt | llm
    feedback = chain.invoke({
        "score": state["final_score"],
        "verdict": state["verdict"],
        "missing_elements": missing_elements_str
    })
    
    return {"missing_elements": missing_elements_str, "feedback": feedback.content}


def should_continue(state: AnalysisState):
    """Determines if the graph should continue or end due to an error."""
    return "end" if state.get("error") else "continue"


workflow = StateGraph(AnalysisState)

workflow.add_node("parse_jd", parse_jd_node)
workflow.add_node("hard_match", hard_match_node)
workflow.add_node("semantic_match", semantic_match_node)
workflow.add_node("scoring", scoring_node)
workflow.add_node("feedback", feedback_node)

workflow.set_entry_point("parse_jd")
workflow.add_edge("parse_jd", "hard_match")
workflow.add_edge("hard_match", "semantic_match")
workflow.add_edge("semantic_match", "scoring")
workflow.add_edge("scoring", "feedback")
workflow.add_edge("feedback", END)


app = workflow.compile()



st.set_page_config(page_title="Automated Resume Relevance Checker", page_icon="📄")
st.title("📄 Automated Resume Relevance Checker")
st.markdown("Upload a Job Description and a Resume to get an instant relevance score and feedback.")


col1, col2 = st.columns(2)

with col1:
    st.header("Job Description")
    jd_file = st.file_uploader("Upload JD (.txt)", type=["txt"])
    jd_text_area = st.text_area("Or paste the Job Description here", height=300)

with col2:
    st.header("Candidate Resume")
    resume_file = st.file_uploader("Upload Resume (.pdf, .docx)", type=["pdf", "docx"])

analyze_button = st.button("Analyze Resume", type="primary")

if analyze_button:
    
    jd_text = ""
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
    elif jd_text_area:
        jd_text = jd_text_area
    else:
        st.error("Please upload or paste the Job Description.")
        st.stop()

    if not resume_file:
        st.error("Please upload a Resume.")
        st.stop()

    
    resume_text = ""
    file_bytes = resume_file.getvalue()
    if resume_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(file_bytes)
    elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(file_bytes)
    
    if not resume_text:
        st.error("Could not extract text from the resume. The file might be corrupted or empty.")
        st.stop()
    
    with st.spinner("Analyzing... This may take a moment."):
        initial_state = {"jd_text": jd_text, "resume_text": resume_text}
        result_state = app.invoke(initial_state)

    if result_state.get("error"):
        st.error(f"An error occurred during analysis: {result_state['error']}")
    else:
        st.success("Analysis Complete!")
        
        
        st.header("Analysis Results")
        
        score = result_state.get("final_score", 0)
        verdict = result_state.get("verdict", "N/A")
        
       
        st.metric(label="Relevance Score", value=f"{score}%", delta=verdict)
        st.progress(score)
        
        st.subheader("Breakdown:")
        expander = st.expander("Show detailed scores and feedback")
        with expander:
            col_a, col_b = st.columns(2)
            col_a.info(f"**Keyword Match Score:** {result_state.get('hard_match_score', 0.0):.1f}%")
            col_b.info(f"**Semantic Fit Score:** {result_state.get('semantic_score', 0.0):.1f}%")
            
            st.markdown("**Missing Skills / Elements:**")
            st.warning(result_state.get("missing_elements", "N/A"))

            st.markdown("**Personalized Feedback for Student:**")
            st.success(result_state.get("feedback", "No feedback generated."))



