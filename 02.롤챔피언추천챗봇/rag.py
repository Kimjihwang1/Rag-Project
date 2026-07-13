from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

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

########################################################################
# 웹 검색 Tool (최신 정보용)
########################################################################

duck = DuckDuckGoSearchRun()

# 이 키워드가 질문에 포함되면 문서 검색 대신 웹 검색으로 우회
REALTIME_KEYWORDS = ["최신", "패치", "오늘", "현재", "요즘", "밸런스 패치", "너프", "버프", "신규 챔피언"]


def is_realtime_question(question: str) -> bool:
    return any(kw in question for kw in REALTIME_KEYWORDS)


# 검색어 생성기 (Query Transformation) 프롬프트 및 체인
query_gen_prompt = ChatPromptTemplate.from_template(
"""
사용자의 질문을 웹 검색엔진(구글 등)에 입력하여 가장 최신의 정확한 정보를 얻기에 적합한 검색 키워드로 변환해 주세요.
불필요한 문장 성분은 지우고 핵심 단어 위주의 검색어 1개만 완성해서 답변하세요. 따옴표나 부가 설명은 절대 쓰지 마세요.

질문: {question}
검색어:
"""
)
query_gen_chain = query_gen_prompt | llm | StrOutputParser()


web_prompt = ChatPromptTemplate.from_template(
"""
당신은 리그 오브 레전드 초보자를 위한 챔피언 추천 및 게임 가이드 AI입니다.

아래는 웹 검색 결과입니다. 이 내용을 참고하여 사용자 질문에 친절하고 이해하기 쉽게 답변하세요.
검색 결과에 없는 내용은 지어내지 말고, 검색 결과가 질문과 관련 없다면 정보를 찾지 못했다고 솔직히 말하세요.

이전 대화
{history}

검색 결과
{search_result}

질문
{question}
"""
)

web_chain = web_prompt | llm | StrOutputParser()


from langchain_core.documents import Document

def web_search_answer(question: str, history: str = "") -> str:
    # 검색용 쿼리 최적화 수행 (Query Transformation)
    search_query = query_gen_chain.invoke({"question": question}).strip()
    search_query = search_query.replace('"', '').replace("'", "")
    
    # "리그 오브 레전드" 키워드를 덧붙여 검색 정확도 확보
    full_search_query = f"리그 오브 레전드 {search_query}"
    
    print(f"=== 생성된 검색 쿼리: {full_search_query} ===")
    search_result = duck.run(full_search_query)

    print("=== 웹 검색 결과 ===")
    print(search_result[:300])
    print("---")

    answer = web_chain.invoke({
        "history": history,
        "search_result": search_result,
        "question": question
    })

    # 웹 검색 결과를 Document 객체로 포장하여 리턴
    web_docs = [Document(page_content=search_result, metadata={"source": "웹 검색 결과"})]
    return answer, web_docs


########################################################################
# 기존 RAG (문서 기반 챔피언 정보)
########################################################################

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
12. 특정 챔피언 이름이 질문에 포함된 경우, 반드시 그 챔피언 자신의 "### 챔피언명:" 항목만을 근거로 답하세요. 다른 챔피언의 "상대하기 쉬운/어려운 챔피언" 필드를 가져와 그 챔피언과 연관지어 설명하지 마세요.
13. 해당 챔피언의 "상대하기 쉬운/어려운 챔피언" 필드에 구체적인 챔피언 이름이 없고 유형(예: "지속 능력이 뛰어난 AP 브루저")만 적혀 있다면, 그 유형 그대로 답변하세요. 없는 챔피언 이름을 만들어내거나 다른 챔피언 정보와 억지로 연결해 이유를 지어내지 마세요.

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
    # 최신 정보(패치, 밸런스 등) 질문이면 웹 검색으로 우회
    if is_realtime_question(question):
        return web_search_answer(question, history)

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

    return answer, docs