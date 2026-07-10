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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("질문을 입력하세요. (예: 정글 처음 하는데 뭐가 쉬워?)")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            answer = ask(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})