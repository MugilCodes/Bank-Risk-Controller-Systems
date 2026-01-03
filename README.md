🏦 Bank Risk Controller System
Loan Default Prediction & Intelligent Banking Assistant (RAG + LLM)
📌 Project Overview

The Bank Risk Controller System is an end-to-end Data Science & AI application designed to help financial institutions assess loan default risk and assist customers through an intelligent chatbot.

This system combines:

Machine Learning for loan default prediction
Exploratory Data Analysis (EDA) for business insights
Streamlit Web Application for interactivity
Retrieval Augmented Generation (RAG) powered by LLMs for banking queries

🎯 Problem Statement

Banks face major challenges due to:

Increasing loan defaults
Manual and subjective risk assessment
Lack of scalable customer support systems

Traditional systems:

Are rule-based
Do not adapt to changing data
Cannot understand document-based banking policies

🚀 Solution

This project delivers a smart, automated risk management system that:
Predicts whether a customer is High Risk / Low Risk
Provides real-time insights via EDA dashboards
Answers banking policy queries using a document-aware chatbot

🧠 Key Features
🔹 Loan Default Prediction

Predicts probability of loan default
Uses historical loan application data
Trained using advanced ML models

🔹 Exploratory Data Analysis (EDA)

Interactive charts
Risk distribution analysis
Feature-wise default insights

🔹 Intelligent Banking Chatbot (RAG)

Uses bank policy PDFs
Context-aware responses
Avoids hallucinations
Powered by LLaMA & LangChain

🔹 Streamlit Web Application

User-friendly interface
Real-time predictions
Integrated chatbot
Modular design


📊 Dataset Description

Source:  Loan Dataset
Records: [14,13,700]
Features: [158]
Target Variable:
TARGET = 1 → Loan Default
TARGET = 0 → No Default

Important Features:

AGE
BIRTH_YEAR
YEARS_EMPLOYED
AMT_INCOME_TOTAL
AMT_CREDIT
CNT_CHILDREN
EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
CODE_GENDER
NAME_CONTRACT_STATUS

🧹 Data Preprocessing

Handled missing values
Converted negative day values to:
Age in years
Employment years
Feature Engineering:
Birth year extraction
Children count binning
Encoding categorical variables
Class imbalance handling
Feature alignment to avoid prediction mismatch

📈 Exploratory Data Analysis (EDA)

EDA was conducted to understand:
Default rate by age group
Income vs credit amount risk
Gender-wise default distribution
Impact of external risk scores
Employment duration vs default

📌 Outcome: Identified key predictors influencing loan default.

🤖 Machine Learning Models
Models Tested:

Logistic Regression
Random Forest(Final Model)
Gradient Boosting
LightGBM 
Decision Tree
Extra Tree

Why RF?
High performance on imbalanced data
Faster training
Better generalization

🔮 Prediction Workflow

User enters loan applicant details
Input features are aligned with training features
Model predicts:
Default Probability
Risk Category (High / Low)
✔ Feature name mismatch error resolved
✔ Robust prediction pipeline implemented

💬 Banking Chatbot (RAG System)
Objective:
Provide accurate banking policy answers using documents.
Technologies Used:
LangChain
Sentence Transformers
ChromaDB
LLaMA (Mistral-7B Banking)
Workflow:
Load PDF documents
Split text into chunks
Create embeddings
Store in vector database
Retrieve relevant context
Generate answer using LLM

🖥️ Streamlit Application
Features:

Dataset preview
EDA visualizations
Loan risk prediction form
Interactive chatbot
Real-time responses

🛠️ Tech Stack
Category	Tools
Language	Python
ML	Scikit-learn, LightGBM
Visualization	Matplotlib, Seaborn, Plotly
Web App	Streamlit
LLM	LLaMA (Mistral-7B)
RAG	LangChain, ChromaDB
Embeddings	Sentence Transformers
⚙️ Installation
Clone Repository
git clone https://github.com/your-username/Bank-Risk-Controller-System.git
cd Bank-Risk-Controller-System

Install Dependencies
pip install -r requirements.txt

▶️ Run Application
streamlit run web.py

⚠️ Challenges Faced
Feature mismatch during prediction
High memory usage in Colab
LangChain version conflicts
LLM loading latency

✅ Solutions Implemented

Used model.feature_names_in_
Feature alignment pipeline
Optimized data types
Version-stable imports
Caching and lazy loading

🌟 Results & Impact

Accurate loan default classification
Reduced risk assessment time
Intelligent banking query resolution
Scalable and modular architecture
Real-world financial application

🔮 Future Enhancements

SHAP for explainable AI
Cloud deployment (AWS / Azure)
Multilingual chatbot
Real-time database integration
Role-based access control

