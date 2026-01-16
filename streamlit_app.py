import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool

# 1. 페이지 설정
st.set_page_config(page_title="미래설계 진로 챗봇", layout="centered")
st.title("🎓 미래설계 진로 & 도서 추천 챗봇")
st.caption("Gemini 2.5 Flash와 최신 실시간 검색을 결합한 상담소")

# API 키 설정
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    google_cse_id = st.secrets["GOOGLE_CSE_ID"]
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError as e:
    st.error(f"Secrets 설정 확인 필요: {e}")
    st.stop()

# 2. 모델 및 도구 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.7
)

search = GoogleSearchAPIWrapper(
    google_api_key=google_api_key,
    google_cse_id=google_cse_id
)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="진로, 베스트셀러 도서, 전공 정보 등을 검색할 때 사용합니다."
    )
]

# 3. 최신 방식의 에이전트 생성
prompt = hub.pull("hwchase17/react")  # 표준 ReAct 프롬프트 다운로드
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 4. 채팅 UI 구현
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("관심 있는 진로나 전공을 말씀해 주세요!"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("최신 정보를 찾는 중입니다..."):
            # 시스템 지침을 질문과 결합
            query = f"사용자는 학생입니다. 다음 질문에 대해 친절하게 답하고 관련 도서를 추천해주세요: {user_input}"
            try:
                response = agent_executor.invoke({"input": query})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                print(f"Error: {e}")
