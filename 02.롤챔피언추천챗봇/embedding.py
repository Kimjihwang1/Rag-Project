from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent

# 1. 문서 읽기
documents = []

# PDF 파일들
for pdf_file in BASE_DIR.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

# TXT 파일들 (champions.txt, terms.txt)
for txt_file in BASE_DIR.glob("*.txt"):
    loader = TextLoader(str(txt_file), encoding="utf-8")
    documents.extend(loader.load())

print("문서 개수:", len(documents))

# 2. 문서 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)
print("청크 개수", len(docs))

# 3. 임베딩
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 4. 크로마 DB 생성
DB_PATH = BASE_DIR / "chroma_db"
db = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=str(DB_PATH)
)

print("Vector DB 저장 완료")