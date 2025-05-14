import json
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_vector_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = "./chroma_news_db"
    json_path = "./reuters_articles.json"

    if not os.path.exists(persist_dir):
        print("⚙️ Creating new vectorstore...")
        
        with open(json_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        all_chunks = []

        for article in articles:
            full_text = f"{article['title']}\n\n{article['content']}"
            metadata = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "tags": article.get("tags", ""),
                "date": article.get("date", "")
            }
            chunks = text_splitter.create_documents([full_text], metadatas=[metadata])
            all_chunks.extend(chunks)

        # Create and persist vectorstore
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        vectorstore.persist()
        print(f"✅ Created new index with {len(all_chunks)} chunks from {len(articles)} articles.")
    else:
        # Load existing vectorstore
        print("♻️ Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
