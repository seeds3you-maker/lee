import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.messages import HumanMessage

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
# 캐싱을 이용해 앱 속도 향상 및 재초기화 방지
@st.cache_resource
def load_tools():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=gemini_api_key,
        temperature=0.7
    )
    search = GoogleSearchAPIWrapper(
        google_api_key=google_api_key, 
        google_cse_id=google_cse_id
    )
    return llm, search

llm, search = load_tools()

# 3. 채팅 세션 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 챗봇 실행 로직
if user_input := st.chat_input("어떤 진로가 고민인가요?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("정보를 검색하고 조언을 생성 중입니다..."):
            try:
                # 1단계: 관련 정보 검색
                search_results = search.run(f"{user_input} 관련 학과 진로 추천 도서")
                
                # 2단계: 프롬프트 생성
                prompt = f"""
                당신은 진로 상담 전문가입니다. 
                사용자의 고민: {user_input}
                
                아래의 검색된 정보를 참고하여 친절하게 상담해주고, 반드시 관련 추천 도서 2권을 제목과 함께 소개해 주세요.
                검색 데이터: {search_results}
                """
                
                # 3단계: Gemini 2.5 Flash 호출
                response = llm.invoke([HumanMessage(content=prompt)])
                answer = response.content
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error("답변 생성 중 오류가 발생했습니다.")
                st.info("API 키 유효성이나 Secrets 설정을 다시 확인해 주세요.")
