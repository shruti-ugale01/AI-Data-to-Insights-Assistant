# AI-Data-to-Insights-Assistant

Enterprise-Grade Analytics with RAG + Streamlit + LLMs

Overview:

The AI Data-to-Insights Assistant is an intelligent analytics platform designed to transform raw datasets into actionable insights.
By combining automated data cleaning, interactive visualizations, query-based exploratory analysis, and RAG-powered natural language responses, this assistant bridges the gap between data science and decision-making.

Upload any CSV — and the system will help you:

✔ Explore the dataset
✔ Visualize patterns and relationships
✔ Query the data using natural language
✔ Generate executive-level insight reports

No coding. No dashboards. Just intelligence.

Key Features
Capability	Description
CSV Upload Support	Upload any structured dataset for instant analysis
 Automatic Data Cleaning	Missing value handling + dynamic type inference
 Smart Visualizations	Histogram, bar chart, box plot, heatmap (auto column detection)
 Explore Mode	Filter data using expressions (Age > 30 and Gender == "Male")
 RAG-Based Querying	Ask questions like: “Which department has the highest salary?”
 Insight Report Generator	AI-generated humans-readable summary of trends & patterns
 Premium UI	Custom gradient theme with glassmorphism + animations
 
User (Streamlit UI)
       │
       ▼
CSV Upload → Pandas → Data Cleaning → Visualization (Plotly)
       │
       ▼
Data → Markdown → Text Split → Sentence Transformers Embeddings → ChromaDB
       │
       ▼
Retriever → LLM (ChatOllama) → Natural Language Insights + Recommendations

Tech Stack
Category	Technology
Framework	Streamlit
Language Model Ollama Llama 3.2
Embeddings Sentence-Transformers (MiniLM-L6-v2)
Vector Store ChromaDB
Visualization	Plotly Express
Data Processing	Pandas
RAG Pipeline LangChain


 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/ai-data-to-insights
cd ai-data-to-insights

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install and pull LLM (Required)
ollama pull llama3.2

4️⃣ Run the application
streamlit run app.py

How to Use:

Open the app in browser (Streamlit auto-opens).

Upload your CSV file.

Explore:

 Overview tab → dataset summary

 Charts → generate auto insights via graphs

 Explore → filter, detect outliers

 Ask AI → natural language questions

 Insight Report → one-click dataset intelligence

 Example Questions You Can Ask

"Which role has the highest attrition?"
"Is salary correlated with experience?"
"Which demographic group appears dominant?"
"What is the main pattern in this dataset?"
te about Applied AI, Data Intelligence, and User-Centric Automation.
