import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool

# 1. 페이지 설정
st.set_page_config(page_title="진로 & 도서 추천 챗봇", layout="centered")
st.title("🎓 미래설계 진로 챗봇")
st.caption("Gemini 2.5 Flash 기반의 지능형 상담소")

# API 키 설정 (보안)
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    google_cse_id = st.secrets["GOOGLE_CSE_ID"]
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError as e:
    st.error(f"Streamlit Secrets 설정에 {e}가 누락되었습니다.")
    st.stop()

# 2. 도구 및 모델 설정
@st.cache_resource
def init_agent():
    # 최신 Gemini 2.5 Flash 모델
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=gemini_api_key,
        temperature=0.7
    )
    
    # 구글 검색 도구
    search = GoogleSearchAPIWrapper(
        google_api_key=google_api_key, 
        google_cse_id=google_cse_id
    )
    
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="진로 정보, 학과 정보, 최신 도서 및 베스트셀러를 찾을 때 사용합니다."
        )
    ]
    
    # ReAct 프롬프트 로드 및 에이전트 생성
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent_executor = init_agent()

# 3. 채팅 UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("진로 고민이나 관심 있는 분야를 알려주세요!"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("정보를 분석 중입니다..."):
            # 구체적인 답변 가이드라인 제공
            prompt_query = f"""당신은 진로 상담가입니다. 
            사용자의 질문: '{user_input}'에 대해 답변하고, 
            Search 도구를 사용하여 관련된 최신 추천 도서 2~3권을 반드시 포함해서 답변하세요."""
            
            try:
                response = agent_executor.invoke({"input": prompt_query})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("답변 생성 중 일시적인 오류가 발생했습니다.")
                st.info("로그를 확인하거나 잠시 후 다시 시도해 주세요.")
