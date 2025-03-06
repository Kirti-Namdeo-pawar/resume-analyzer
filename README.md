

https://github.com/user-attachments/assets/4babf10e-0cb6-48e8-8a9d-83dc2570b8b0

Resume Analyzer

A machine learning-powered resume analyzer that classifies resumes into job categories using NLP techniques.

📌 Features

Predicts job categories from resumes.

Uses NLP techniques to process and analyze text.

Supports multiple job categories such as IT, Healthcare, Finance, etc.

Handles imbalanced data using SMOTE.

Provides insights into candidate skills and job suitability.

🛠 Tech Stack

Python 3.12

Pandas, NumPy

Scikit-learn, Imbalanced-learn

Natural Language Processing (NLP)

Machine Learning (Logistic Regression, Random Forest, etc.)

🚀 Installation

Clone the Repository:

git clone https://github.com/your-username/resume-analyzer.git
cd resume-analyzer

Create Virtual Environment (Optional but Recommended):

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

📂 Dataset

The dataset contains resumes labeled by job category.

Format: CSV

Columns: ID, Resume_str, Resume_html, Category

Sample job categories: IT, Healthcare, Finance, HR, Engineering, etc.

🎯 Usage

1️⃣ Train the Model

python train_model.py

2️⃣ Predict Job Category

python predict.py --resume "path/to/resume.pdf"

3️⃣ Evaluate Model Performance

python evaluate.py

# resume-analyzer
