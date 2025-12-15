import os
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")

# Initialize Flask
app = Flask(__name__)

@app.route("/")
def index():
    # Serve your HTML file
    return send_file("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    url = data.get("url")
    question = data.get("question")

    if not url or not question:
        return jsonify({"error": "URL and question are required"}), 400

    # 1. Load webpage
    loader = WebBaseLoader(url)
    docs = loader.load()

    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Embeddings + vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    # 5. Prompt
    prompt = ChatPromptTemplate.from_template(
        "Use this context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # 6. Chain
    chain = ({"context": retriever, "question": lambda x: x} | prompt | llm | StrOutputParser())

    # 7. Run query
    answer = chain.invoke(question)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
