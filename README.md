# fastapi_backend

# SmartDocs AI Backend

This repository hosts the backend for **SmartDocs AI**, a platform that allows users to upload PDF or DOC files and query their contents using AI-powered tools. The backend is built with **FastAPI** and integrates **LangChain**, **Hugging Face**, **MongoDB**, and **OpenAI** for seamless document processing and query handling.

## 🚀 Live Demo

Check out the live application: [SmartDocs AI](https://www.smartdocsai.site/)

---

## 🛠️ Features

- Upload PDF documents and extract their text.
- Query uploaded documents for specific content.
- AI embeddings powered by Hugging Face.
- Pinecone integration for vector storage and similarity search.
- FastAPI for a high-performance backend.

---

## 🏗️ Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **AI & Embeddings**: [LangChain](https://www.langchain.com/), [Hugging Face](https://huggingface.co/), [OpenAI](https://platform.openai.com/)
- **Database**: [MongoDB](https://www.mongodb.com/)
- **Vector Storage**: [Pinecone](https://www.pinecone.io/)

---

## 📋 Prerequisites

Before you start, ensure you have the following installed:

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/installation/)
- [MongoDB](https://www.mongodb.com/try/download/community)
- [Pinecone Account](https://www.pinecone.io/)
- [Hugging Face Account](https://huggingface.co/)
- [OpenAI API Key](https://platform.openai.com/signup/)

---

## ⚙️ Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Prasoon321/fastapi_backend.git
   cd smartdocs-backend
   ```

python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt

OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
MONGO_URI=your_mongo_connection_string
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment

uvicorn main:app --reload

The application will be available at: http://127.0.0.1:8000

```

2. **Api Endpoint**:
```

{
"message": "PDF uploaded and content stored successfully"
}

{
"answer": ["Relevant content from the document"]
}

.
├── main.py # FastAPI application
├── .env # Environment variables
├── requirements.txt # Python dependencies
└── README.md # Project documentation

```

```
