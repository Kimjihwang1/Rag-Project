from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_loader import init_custom_llm
llm = init_custom_llm()

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

db = Chroma(
    embedding_function=embedding,
    persist_directory=str(DB_PATH)
)

retriever = db.as_retriever(search_kwargs={"k": 100})

prompt = ChatPromptTemplate.from_template(
"""
당신은 리그 오브 레전드 초보자를 위한 챔피언 추천 및 게임 가이드 AI입니다.
아래 문서를 참고하여 친절하고 이해하기 쉽게 답변하세요.

규칙:
1. 반드시 아래 "문서"에 등장하는 챔피언 이름만 언급하세요.
2. 문서에 없는 챔피언 이름을 절대 지어내지 마세요.
3. 문서에서 질문에 대한 답을 찾을 수 없으면, 지어내지 말고
   "제공된 정보 안에서는 확인할 수 없습니다"라고 답하세요.
4. "OO를 카운터하는 챔피언", "OO의 천적", "OO를 상대하기 어려운 챔피언"은
   문서의 "상대하기 어려운 챔피언" 항목(OO 입장에서 상대하기 어려운 상대)과 같은 의미입니다.
5. "OO가 상대하기 쉬운 챔피언", "OO가 이기기 쉬운 상대"는
   문서의 "상대하기 쉬운 챔피언" 항목과 같은 의미입니다.

문서
{context}

질문
{question}
"""
)

chain = prompt | llm | StrOutputParser()

def format_docs(docs):
    result = ""
    for doc in docs:
        result += doc.page_content
        result += "\n\n"
    return result

def ask(question):
    docs = retriever.invoke(question)
    print("=== 검색된 문서 ===")
    for i, doc in enumerate(docs):
        print(f"[{i}] {doc.page_content[:150]}")
        print("---")
    context = format_docs(docs)
    answer = chain.invoke({"context": context, "question": question})
    return answer