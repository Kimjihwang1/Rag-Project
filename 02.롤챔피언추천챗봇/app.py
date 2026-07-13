import streamlit as st
from rag import ask

st.set_page_config(
    page_title="롤 챔피언 추천 챗봇",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 롤 초보자를 위한 챔피언 추천 & 가이드 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

########################################################################
# 사이드바 - 예시 질문 버튼 + 대화 초기화
########################################################################

with st.sidebar:
    st.subheader("💡 예시 질문")
    examples = [
        "정글 처음 하는데 뭐가 쉬워?",
        "제드 카운터 챔피언 알려줘",
        "탑 초보자용 챔피언 추천해줘",
        "서포터 중에 쉬운 챔피언은?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex

    st.divider()

    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

########################################################################
# 대화 내용 렌더링 (아바타 포함)
########################################################################

if not st.session_state.messages:
    st.info("👋 챔피언 추천이나 카운터 정보를 물어보세요! 예: '정글 처음인데 뭐가 쉬워?'")

for message in st.session_state.messages:
    avatar = "🎮" if message["role"] == "assistant" else "🧑"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

########################################################################
# 질문 입력 처리 (사이드바 버튼 클릭 or 직접 입력)
########################################################################

question = st.chat_input("질문을 입력하세요. (예: 정글 처음 하는데 뭐가 쉬워?)")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    # 이전 대화 생성
    history = ""
    for msg in st.session_state.messages[:-1]:   # 방금 질문 제외
        role = "사용자" if msg["role"] == "user" else "AI"
        history += f"{role}: {msg['content']}\n"

    with st.chat_message("assistant", avatar="🎮"):
        with st.spinner("생각하는 중..."):
            answer, docs = ask(question, history)

        st.markdown(answer)

        # 검색된 문서(근거) 펼쳐보기
        if docs:
            is_web = any(doc.metadata.get("source") == "웹 검색 결과" for doc in docs)
            if is_web:
                with st.expander("🌐 참고한 실시간 웹 검색 결과 보기"):
                    for doc in docs:
                        st.markdown(doc.page_content)
            else:
                with st.expander("📄 참고한 챔피언 문서 보기"):
                    for i, doc in enumerate(docs):
                        st.text(f"[{i}] {doc.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})