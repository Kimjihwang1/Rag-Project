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

이전 대화
{history}

아래 문서를 참고하여 친절하고 이해하기 쉽게 답변하세요.

규칙:
1. 반드시 아래 "문서"에 등장하는 챔피언 이름만 언급하세요.
2. 문서에 없는 챔피언 이름을 절대 지어내지 마세요.
3. 문서에 직접적인 답이 없는 경우에도, 문서에 포함된 정보만을 근거로 비교·추천할 수 있습니다 단, 문서에 없는 새로운 챔피언이나 정보를 추가하지 마세요.
4. "OO를 카운터하는 챔피언", "OO의 천적", "OO를 상대하기 어려운 챔피언"은 문서의 "상대하기 어려운 챔피언" 항목(OO 입장에서 상대하기 어려운 상대)과 같은 의미입니다.
5. "OO가 상대하기 쉬운 챔피언", "OO가 이기기 쉬운 상대"는 문서의 "상대하기 쉬운 챔피언" 항목과 같은 의미입니다.
6. 사용자가 "쉬운 챔피언", "초보자용 챔피언", "입문용 챔피언"을 요청하면 문서의 "난이도: 하", "난이도: 최하", 또는 "초보자 팁"을 참고하여 추천하세요.
7. 사용자가 "어려운 챔피언", "고난도 챔피언", "숙련도가 필요한 챔피언"을 요청하면 문서의 "난이도: 상", "난이도: 최상"을 참고하여 추천하세요.
8. 이전 대화에서 언급한 챔피언이나 내용을 참고하여 사용자의 질문 대상을 이해하세요.
9. 이전 대화에서 추천했던 챔피언들 중 하나를 가리키는 질문(예: "그중에", "걔는?", "첫 번째", "둘 중에")이 들어오면 이전 대화를 참고하여 답변하세요.
10. 추천을 요청하면 가능한 경우 2~3명의 챔피언을 추천하세요.
11. 답변의 근거는 반드시 아래 문서입니다.

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



def ask(question, history=""):
    # 현재 질문만 검색
    docs = retriever.invoke(question)

    print("=== 검색된 문서 ===")
    for i, doc in enumerate(docs):
        print(f"[{i}] {doc.page_content[:150]}")
        print("---")

    context = format_docs(docs)

    answer = chain.invoke({
        "history": history,
        "context": context,
        "question": question
    })

    return answer