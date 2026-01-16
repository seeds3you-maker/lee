import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import langchainhub as hub
# 임포트 경로를 더 구체적으로 명시하여 에러 방지
from langchain.agents.agent import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool

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

# 2. 모델 및 도구 설정
@st.cache_resource
def init_agent():
    # 최신 안정화 모델 Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=gemini_api_key,
        temperature=0.7
    )
    
    # 구글 검색 엔진 설정
    search = GoogleSearchAPIWrapper(
        google_api_key=google_api_key, 
        google_cse_id=google_cse_id
    )
    
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="진로 정보, 베스트셀러 도서 정보를 검색할 때 사용합니다."
        )
    ]
    
    # ReAct 프롬프트 로드
    prompt = hub.pull("hwchase17/react")
    
    # 에이전트 생성 (최신 방식)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

try:
    agent_executor = init_agent()
except Exception as e:
    st.error(f"에이전트 초기화 중 오류 발생: {e}")
    st.stop()

# 3. 채팅 UI 구성
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("어떤 진로가 고민인가요?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("최신 도서와 정보를 검색 중입니다..."):
            prompt_query = f"사용자 질문: {user_input}. 관련 진로 도서를 검색하여 추천하고 상담해줘."
            try:
                # 최신 invoke 방식 사용
                response = agent_executor.invoke({"input": prompt_query})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("답변 생성 과정에서 오류가 발생했습니다.")
                st.info("검색 API 할당량이나 키 설정을 확인해 주세요.")
