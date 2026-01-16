import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory

# 페이지 설정 및 UI
st.set_page_config(page_title="미래설계 진로 챗봇", layout="centered")
st.title("🎓 미래설계 진로 & 도서 추천 챗봇")
st.caption("여러분의 꿈을 위해 gemini-2.5-flash가 최신 정보를 바탕으로 조언해 드립니다.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# API 키 설정 (Streamlit Secrets)
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    google_cse_id = st.secrets["GOOGLE_CSE_ID"]
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError as e:
    st.error(f"Secrets 설정이 누락되었습니다: {e}")
    st.stop()

# 1. 모델 설정 (최신 gemini-2.5-flash 사용)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.7
)

# 2. 도구 설정 (실시간 도서 및 진로 정보 검색을 위한 Google Search)
search = GoogleSearchAPIWrapper(
    google_api_key=google_api_key,
    google_cse_id=google_cse_id
)

tools = [
    Tool(
        name="CareerBookSearch",
        func=search.run,
        description="특정 진로 분야의 최신 도서, 베스트셀러, 추천 도서 정보를 검색할 때 사용합니다."
    )
]

# 3. 에이전트 초기화
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=st.session_state.memory,
    handle_parsing_errors=True
)

# 시스템 프롬프트 정의
SYSTEM_PROMPT = """너는 학생들의 진로를 상담해주는 전문 컨설턴트야.
사용자의 관심사나 전공에 맞춰 구체적인 로드맵을 제시해주고, 
반드시 'CareerBookSearch' 도구를 사용하여 해당 분야의 최신 베스트셀러나 평점이 좋은 도서를 찾아 추천해줘.
답변은 친절하고 격려하는 말투로 작성해줘."""

# 채팅 인터페이스 구현
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("진로에 대해 궁금한 점이나 관심 분야를 말씀해 주세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("최신 정보를 검색하며 답변을 생성 중입니다..."):
            full_prompt = f"{SYSTEM_PROMPT}\n\n사용자 질문: {prompt}"
            response = agent_chain.run(input=full_prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})