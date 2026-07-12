# 🎮 롤 챔피언 추천 RAG 챗봇

LLM과 RAG(Retrieval-Augmented Generation)를 활용하여
사용자의 질문에 맞는 리그 오브 레전드(LoL) 챔피언을 추천하는 챗봇입니다.

---

## 프로젝트 소개

이 프로젝트는 LangChain과 ChromaDB를 이용하여
챔피언 정보를 벡터 데이터베이스에 저장하고,
질문과 가장 관련 있는 문서를 검색한 뒤 LLM이 답변을 생성하는
RAG(Retrieval-Augmented Generation) 방식으로 구현했습니다.

예를 들어 다음과 같은 질문이 가능합니다.

- 미드 초보자에게 추천할 챔피언은?
- 제드를 상대하기 좋은 챔피언은?
- 원거리 딜러 추천해줘
- 쉬운 정글 챔피언 알려줘

---

## 프로젝트 구조

```
02.롤챔피언추천챗봇
│
├── app.py              # Streamlit 실행
├── rag.py              # RAG 파이프라인
├── embedding.py        # 문서 임베딩 생성
├── llm_loader.py       # LLM 로드
├── champions.txt       # 챔피언 데이터
├── terms.txt           # 롤 용어 데이터
└── chroma_db/          # Chroma 벡터DB
```

---

## 사용 기술

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embedding
- Ollama (LLM)

---

## RAG 동작 과정

```
사용자 질문
      │
      ▼
ChromaDB에서 관련 문서 검색
      │
      ▼
검색된 문서를 Prompt에 포함
      │
      ▼
LLM이 검색 결과를 기반으로 답변 생성
```

---

## 주요 기능

- 챔피언 추천
- 포지션별 검색
- 난이도 기반 추천
- 챔피언 카운터 검색
- 롤 용어 설명
- RAG 기반 답변 생성

---

## 개발 과정에서 해결한 문제

### 1. Hallucination(환각) 문제

존재하지 않는 챔피언을 생성하는 문제가 발생하여

- 문서에 존재하는 챔피언만 답변
- 없는 경우 "제공된 정보 안에서는 확인할 수 없습니다."

라는 규칙을 Prompt에 추가했습니다.

---

### 2. Retrieval 성능 개선

질문에 필요한 문서를 검색하지 못하는 문제가 발생했습니다.

원인

- 벡터 검색이 특정 단어에 치우쳐 필요한 문서를 찾지 못함

해결

- Retriever의 k 값을 조정하여 검색 범위를 확대
- 검색 결과를 직접 출력하며 디버깅

---

### 3. ChromaDB 관리

프로젝트 폴더와 상위 폴더에
ChromaDB가 각각 생성되어
최신 데이터가 반영되지 않는 문제가 발생했습니다.

기존 DB를 삭제한 후
Embedding을 다시 생성하여 해결했습니다.

---

## 실행 방법

### 1. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 임베딩 생성

```bash
python embedding.py
```

### 3. 실행

```bash
streamlit run app.py
```

---

## 예시 질문

- 미드 입문용 챔피언 추천
- 제드를 카운터치는 챔피언 알려줘
- 탑에서 쉬운 챔피언 추천
- 초보 정글러 추천
- 스노우볼이 뭐야?
- 라인전이 뭐야?

---

## 프로젝트를 통해 배운 점

- LangChain을 이용한 RAG 파이프라인 구성
- ChromaDB를 활용한 벡터 검색
- Prompt Engineering의 중요성
- Retrieval 결과를 직접 확인하며 디버깅하는 방법
- Hallucination을 줄이기 위한 Prompt 설계