import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.core.messages import HumanMessage, SystemMessage

# 1. 페이지 설정
st.set_page_config(page_title="진로 & 도서 추천 챗봇", layout="centered")
st.title("🎓 미래설계 진로 챗봇")
st.caption("Gemini 2.5 Flash 기반의 실시간 상담소")

# API 키 보안 호출
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    google_cse_id = st.secrets["GOOGLE_CSE_ID"]
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError as e:
    st.error(f"Secrets 설정 확인 필요: {e} 키가 누락되었습니다.")
    st.stop()

# 2. 모델 및 검색 도구 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=gemini_api_key,
    temperature=0.7
)

search = GoogleSearchAPIWrapper(
    google_api_key=google_api_key, 
    google_cse_id=google_cse_id
)

# 3. 채팅 세션 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 챗봇 로직
if user_input := st.chat_input("진로에 대해 무엇이든 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("최신 정보를 검색하며 답변을 준비 중입니다..."):
            # RAG: 구글 검색을 통한 정보 보강
            try:
                search_query = f"{user_input} 관련 학과 진로 추천 도서 베스트셀러"
                search_results = search.run(search_query)
                
                # 프롬프트 구성
                context_prompt = f"""
                당신은 친절한 진로 상담가입니다. 
                아래 검색된 최신 정보를 바탕으로 학생의 질문에 답변하고, 관련 도서를 2~3권 추천해주세요.
                검색 결과: {search_results}
                
                사용자 질문: {user_input}
                """
                
                # Gemini 2.5 Flash 답변 생성
                response = llm.invoke([HumanMessage(content=context_prompt)])
                answer = response.content
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error("답변 생성 중 오류가 발생했습니다. API 키와 설정을 확인해주세요.")
                st.info(f"상세 에러: {e}")
