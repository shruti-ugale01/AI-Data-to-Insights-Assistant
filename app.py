import streamlit as st
import pandas as pd
import plotly.express as px

# LangChain + RAG Modules
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# --------------------------- PAGE SETTINGS ---------------------------
st.set_page_config(page_title="Enterprise Analytics AI", layout="wide")


# --------------------------- THEME CSS ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: #DDE7FF !important;
}

body {
    background: radial-gradient(circle at top, #0E1525, #050A12 60%);
}

.main-title {
    font-size: 45px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg,#00E7FF,#4A7BFF,#4EF7C3);
    -webkit-background-clip: text;
    color: transparent;
    margin-bottom:-5px;
    text-shadow:0px 0px 12px rgba(0,200,255,0.5);
}

.sub-text {
    font-size:18px;text-align:center;color:#A7B7D9;margin-bottom:30px;
}

/* Upload */
[data-testid="stFileUploader"] {
    background:rgba(255,255,255,0.06);
    border:2px dashed #00D1FF;
    border-radius:12px;
    padding:18px;
}

/* Cards */
.card {
    padding:18px;
    border-radius:18px;
    background:rgba(255,255,255,0.05);
    backdrop-filter:blur(15px);
    border:1px solid rgba(255,255,255,0.12);
}

/* Buttons */
.stButton>button {
    background:linear-gradient(90deg,#006CFF,#00E5FF);
    color:white;font-weight:600;border-radius:10px;
    padding:14px;transition:.3s;border:none;
}
.stButton>button:hover {
    transform:scale(1.07);
    box-shadow:0px 0px 20px #00E7FF;
}

/* Chat bubble */
.chat-box {
    background:rgba(0,153,255,0.15);
    padding:16px;
    border-radius:14px;
    border-left:4px solid #00E7FF;
    font-size:18px;
    margin-top:10px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size:15px;font-weight:600;background:rgba(255,255,255,0.06);
    padding:12px 20px;color:#ADBFFF;border-radius:10px;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(90deg,#005DFF,#00E7FF);
    color:white !important;
}
</style>
""", unsafe_allow_html=True)



# --------------------------- HEADER ---------------------------
st.markdown("<div class='main-title'> AI Data-to-Insights Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Analyze • Visualize • Query • Generate Insights</div>", unsafe_allow_html=True)



# --------------------------- FILE UPLOAD ---------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    with st.expander("Preview dataset"):
        st.dataframe(df, use_container_width=True)


    # ---------------- DATA CLEANING ----------------
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype in ["float64","int64"]:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)


    # Column classification 
    id_like = ["id","employee_id","empid","index","serial"]
    numeric_cols = [c for c in df_clean.select_dtypes(include=["float64","int64"]).columns if c.lower() not in id_like]
    categorical_cols = [c for c in df_clean.select_dtypes(include=["object"]).columns if c.lower() not in id_like]


    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", " Charts", "Explore Data", "Ask AI", "📄 Insights Report"]
    )


    # -------- Overview --------
    with tab1:
        st.subheader("Dataset Overview")

        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", len(df_clean))
        c2.metric("Columns", df_clean.shape[1])
        c3.metric("Missing Values", df_clean.isnull().sum().sum())

        st.write(df_clean.describe(include="all"))


    # -------- Charting --------
    with tab2:
        st.subheader("Data Visualizations")

        chart_type = st.selectbox("Choose chart style:", ["Histogram","Bar Chart","Box Plot","Correlation Heatmap"])

        if chart_type == "Histogram" and numeric_cols:
            col = st.selectbox("Select column", numeric_cols)
            st.plotly_chart(px.histogram(df_clean, x=col), use_container_width=True)

        elif chart_type == "Bar Chart" and categorical_cols and numeric_cols:
            groupby = st.selectbox("Group by:", categorical_cols)
            val = st.selectbox("Value:", numeric_cols)
            fig = px.bar(df_clean.groupby(groupby)[val].mean().reset_index(), x=groupby, y=val)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot" and numeric_cols and categorical_cols:
            num = st.selectbox("Numeric", numeric_cols)
            cat = st.selectbox("Category", categorical_cols)
            st.plotly_chart(px.box(df_clean, x=cat, y=num), use_container_width=True)

        elif chart_type == "Correlation Heatmap" and len(numeric_cols) > 1:
            st.plotly_chart(px.imshow(df_clean[numeric_cols].corr(), text_auto=True), use_container_width=True)
        else:
            st.warning("⚠ Not enough appropriate columns for this chart type.")


    # -------- Search & Filtering (EXPLORE DATA) --------
    with tab3:
        st.subheader("Filter & Explore")

        query = st.text_input("Enter filter query (example: Salary > 50000 and Gender == 'Male')")

        if query:
            try:
                st.write(df_clean.query(query))
            except Exception:
                st.warning("Invalid query syntax. Try something like: Age > 30 and Gender == 'Female'")

        st.write("---")
        st.write("Outlier Detection")

        num = st.selectbox("Select column for anomaly scan", numeric_cols)
        if num:
            z = abs((df_clean[num] - df_clean[num].mean()) / df_clean[num].std())
            st.write(df_clean[z > 2.5])


    # -------- RAG Q&A --------
    # Build RAG components once so both Ask AI and Report can reuse llm
    text = df_clean.to_markdown(index=False)
    chunks = RecursiveCharacterTextSplitter(chunk_size=800).split_text(text)

    embed = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = Chroma.from_texts(chunks, embed)
    retriever = vector.as_retriever()

    llm = ChatOllama(model="llama3.2", temperature=0.2)

    prompt = ChatPromptTemplate.from_template("""
    You must answer using ONLY the uploaded dataset context.

    Context:
    {context}

    Question:
    {input}

    Answer professionally:
    """)

    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm,prompt))

    with tab4:
        st.subheader("Ask an AI Question")

        user_question = st.text_input("Ask here:")

        if user_question:
            response = rag_chain.invoke({"input": user_question})
            st.markdown(f"<div class='chat-box'>{response.get('answer','No response')}</div>", unsafe_allow_html=True)


    # -------- Insights Report --------
    with tab5:
        st.subheader("AI Insights Report")

        if st.button("Generate Insights"):
            report_prompt = f"""
            Create a professional insights summary with:

            • Top trends  
            • Key observations  
            • Patterns & anomalies  
            • Business implications  
            • Actionable recommendations  

            Dataset sample:
            {df_clean.head(50).to_markdown()}
            """

            with st.spinner("🔍 Analyzing data..."):
                report = llm.invoke(report_prompt)

            st.markdown(f"<div class='chat-box'>{report.content}</div>", unsafe_allow_html=True)


else:
    st.info("Please upload a CSV file to get started.")
