# ================== IMPORTS ==================
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import joblib
import re
import itertools
import datetime
import tempfile
import pdfplumber

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.llms import LlamaCpp, HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Bank Risk Controller System",
    page_icon="🏦",
    layout="wide"
)

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    data = pd.read_csv("/content/drive/MyDrive/Bank risk/cleaned_data.csv")
    results = pd.read_csv("/content/drive/MyDrive/Bank risk/initial_model_results.csv")
    return data, results

data, model_results = load_data()

# ================== SIDEBAR ==================
st.sidebar.title("🏦 Bank Risk Controller System")

menu = st.sidebar.radio(
    "Navigation",
    [
        "📊 Data",
        "📈 EDA - Visual",
        "🔮 Prediction",
        "💬 Chat with PDF (GenAI)"
    ]
)

# ============================================================
# 📊 DATA PAGE
# ============================================================
if menu == "📊 Data":

    st.title("📊 Dataset & Model Performance")

    st.subheader("🔹 Loan Dataset")
    st.dataframe(data.head(500))
    st.info("Showing first 500 rows only to avoid memory issues")

    st.subheader("🔹 Model Performance Metrics")
    st.dataframe(model_results)

# ============================================================
# 📈 EDA PAGE
# ============================================================
elif menu == "📈 EDA - Visual":

    import plotly.express as px

    st.title("📈 Exploratory Data Analysis (EDA)")

    @st.cache_data
    def load_eda():
        df = pd.read_csv("/content/drive/MyDrive/Bank risk/eda_data.csv")
        df["TARGET_LABEL"] = df["TARGET"].map({
            0: "✅ Non-Defaulter (Good Customer)",
            1: "⚠️ Defaulter (Payment Difficulty)"
        })
        return df

    eda = load_eda()

    st.sidebar.header("🔎 EDA Filters")

    gender_filter = st.sidebar.multiselect(
        "Select Gender",
        eda["CODE_GENDER"].unique(),
        eda["CODE_GENDER"].unique()
    )

    contract_filter = st.sidebar.multiselect(
        "Select Loan Type",
        eda["NAME_CONTRACT_TYPE_x"].unique(),
        eda["NAME_CONTRACT_TYPE_x"].unique()
    )

    target_filter = st.sidebar.multiselect(
        "Select Loan Status",
        eda["TARGET_LABEL"].unique(),
        eda["TARGET_LABEL"].unique()
    )

    age_range = st.sidebar.slider(
        "Select Age Range",
        int(eda["AGE_YEARS"].min()),
        int(eda["AGE_YEARS"].max()),
        (20, 65)
    )

    filtered_df = eda[
        (eda["CODE_GENDER"].isin(gender_filter)) &
        (eda["NAME_CONTRACT_TYPE_x"].isin(contract_filter)) &
        (eda["TARGET_LABEL"].isin(target_filter)) &
        (eda["AGE_YEARS"].between(age_range[0], age_range[1]))
    ]

    eda_section = st.selectbox(
        "📊 Select EDA Topic",
        [
            "Target Distribution",
            "Gender Analysis",
            "Loan Type Analysis",
            "Car & Property Ownership",
            "Family & Dependents",
            "Education vs Default",
            "Income Type Analysis"
        ]
    )

    if eda_section == "Target Distribution":
        fig = px.pie(filtered_df, names="TARGET_LABEL", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_section == "Gender Analysis":
        fig = px.histogram(filtered_df, x="CODE_GENDER", color="TARGET_LABEL", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_section == "Loan Type Analysis":
        fig = px.histogram(filtered_df, x="NAME_CONTRACT_TYPE_x", color="TARGET_LABEL", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_section == "Car & Property Ownership":
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(filtered_df, names="FLAG_OWN_CAR"), use_container_width=True)
        with col2:
            st.plotly_chart(px.pie(filtered_df, names="FLAG_OWN_REALTY"), use_container_width=True)

    elif eda_section == "Family & Dependents":
        fig = px.box(filtered_df, x="TARGET_LABEL", y="CNT_FAM_MEMBERS")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_section == "Education vs Default":
        fig = px.histogram(filtered_df, y="NAME_EDUCATION_TYPE", color="TARGET_LABEL", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_section == "Income Type Analysis":
        fig = px.histogram(filtered_df, y="NAME_INCOME_TYPE", color="TARGET_LABEL", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🔮 PREDICTION PAGE  ✅ FIXED
# ============================================================
elif menu == "🔮 Prediction":

    model = joblib.load("/content/drive/MyDrive/Bank risk/best_lightgbm .pkl")
    FEATURES = model.feature_names_in_

    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Income", 50000, 5000000, 300000)
    credit = st.number_input("Loan Amount", 50000, 5000000, 600000)
    annuity = st.number_input("Annuity", 10000, 500000, 25000)
    goods_price = st.number_input("Goods Price", 50000, 5000000, 550000)
    employed_years = st.slider("Years Employed", 0, 40, 5)
    children = st.selectbox("Children", [0, 1, 2, 3])
    family_members = st.slider("Family Members", 1, 10, 3)
    ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.6)
    ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.55)

    education = st.selectbox(
        "Education",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary"
        ]
    )

    if st.button("Predict"):

        X = np.zeros((1, len(FEATURES)))
        feature_index = {f: i for i, f in enumerate(FEATURES)}

        def set_val(name, value):
            if name in feature_index:
                X[0, feature_index[name]] = value

        set_val("AGE", age)
        set_val("BIRTH_YEAR", datetime.datetime.now().year - age)
        set_val("YEARS_EMPLOYED", employed_years)
        set_val("AMT_INCOME_TOTAL", income)
        set_val("AMT_CREDIT_x", credit)
        set_val("AMT_ANNUITY_x", annuity)
        set_val("AMT_GOODS_PRICE_x", goods_price)
        set_val("CNT_FAM_MEMBERS", family_members)
        set_val("EXT_SOURCE_2", ext2)
        set_val("EXT_SOURCE_3", ext3)

        if gender == "Male":
            set_val("CODE_GENDER_M", 1)

        set_val(f"NAME_EDUCATION_TYPE_{education}", 1)

        if children == 1:
            set_val("CNT_CHILDREN_BIN_1", 1)
        if children >= 2:
            set_val("CNT_CHILDREN_BIN_2+", 1)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk | Default Probability: {probability:.2%}")
        else:
            st.success(f"✅ Low Risk | Default Probability: {probability:.2%}")

# ============================================================
# 💬 CHAT WITH PDF PAGE  (CHAT HISTORY ADDED)
# ============================================================
elif menu == "💬 Chat with PDF (GenAI)":

    st.title("🏦 Bank Risk Assistant (RAG Chatbot)")

    PDF_PATH = "/content/drive/MyDrive/Bank risk/data"
    MODEL_PATH = "/content/drive/MyDrive/Bank risk/Mistral-7B-Banking-v2-Q5_K_M.gguf"

    # ===================== VECTORSTORE =====================
    @st.cache_resource
    def load_vectorstore():
        loader = PyPDFDirectoryLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        embeddings = SentenceTransformerEmbeddings(
            model_name="aminhaeri/risk-embed"
        )

        return Chroma.from_documents(chunks, embeddings)

    # ===================== LLM =====================
    @st.cache_resource
    def load_llm():
        return LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.2,
            max_tokens=2048,
            top_p=1,
            n_ctx=4096,
            verbose=False
        )

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = load_llm()

    # ===================== PROMPT (UNCHANGED) =====================
    template = '''
    <|context|>
    You are an Chatbot to resolve the bank customers queries based on the query and the context provided.
    Please be truthful and give direct answer.
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    '''

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ===================== CHAT HISTORY STATE =====================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ===================== DISPLAY FULL CONVERSATION =====================
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ===================== USER INPUT =====================
    user_input = st.chat_input("Ask a banking-related question...")

    if user_input:

        # --- Show user message ---
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- Generate & show assistant response ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(user_input)
                st.markdown(response)

        # --- Save assistant message ---
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

    # ===================== CLEAR CHAT =====================
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()
