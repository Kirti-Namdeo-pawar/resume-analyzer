import streamlit as st
import joblib
import fitz  # PyMuPDF (to extract text from PDFs)
import re

# Set Streamlit Page Config
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load model & vectorizer
model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Read PDF
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# --- UI Styling ---
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #ff4b4b;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 20px;
            color: #666;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-box {
            border: 2px dashed #ff4b4b;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            background-color: #fff;
        }
        .result-box {
            padding: 15px;
            border-radius: 8px;
            background-color: #e8f5e9;
            border-left: 5px solid #2e7d32;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Layout ---
st.markdown('<h1 class="main-title">📄 Resume Category Classifier</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Upload a PDF or Enter Resume Text to Predict the Job Category</h3>', unsafe_allow_html=True)

# --- Upload Section ---
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Resume (PDF Only)", type=["pdf"])
st.markdown('</div>', unsafe_allow_html=True)

resume_text = st.text_area("Or Paste Resume Text Here:")

# Extract text if PDF is uploaded
if uploaded_file is not None:
    with st.spinner("Extracting text from PDF... ⏳"):
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted successfully!")

# --- Prediction Button ---
if st.button("🔍 Predict Job Category"):
    if resume_text.strip() == "":
        st.warning("⚠️ Please upload a PDF or enter resume text!")
    else:
        cleaned_text = preprocess_text(resume_text)
        input_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(input_vector)[0]

        # Display result in a nicely formatted box
        st.markdown(f'<div class="result-box"><h4>🎯 Predicted Job Category: <b>{prediction}</b></h4></div>', unsafe_allow_html=True)

        # Expandable box to show extracted resume text
        with st.expander("📜 View Extracted Resume Content"):
            st.write(resume_text)
