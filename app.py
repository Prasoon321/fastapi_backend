import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
load_dotenv()
app = FastAPI()
origins = [
    "https://www.smartdocsai.site",  # Your frontend URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
index_name = "docquery"  # Pinecone index name
def extract_text_from_pdf(pdf_file: UploadFile):
    # print("entered pdf text extracter")
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file.file)
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page.extract_text()
            if page_text:
                page_text = page_text.replace("\n", " ").strip()
                text += page_text

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        final_documents = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in final_documents]

        return documents

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")
@app.post("/api/upload-pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    try:
        # print("Uploading PDF")
        if not pdf:
            raise HTTPException(status_code=400, detail="Missing PDF")
        documents = extract_text_from_pdf(pdf)
        huggingface_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        docsearch = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=huggingface_embeddings,
            index_name=index_name
        )
        return JSONResponse(content={"message": "PDF uploaded and content stored successfully"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading PDF: {str(e)}")
@app.post("/api/query-pinecone")
async def query_pinecone(query: str = Form(...)):
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Missing query")
        huggingface_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        docsearch = PineconeVectorStore(
            embedding=huggingface_embeddings,
            index_name=index_name
        )
        relevant_documents = docsearch.similarity_search(query, k=1)
        doc_data = [doc.page_content for doc in relevant_documents]
        return JSONResponse(content={"answer": doc_data})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error querying Pinecone: {str(e)}")
@app.get("/")
async def root():
    return {"message": "Hello World"}