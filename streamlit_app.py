import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub  # 이 방식으로 변경하여 에러 방지
from langchain.agents import AgentExecutor, create_react_agent
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
    
    # ReAct 프롬프트 로드 (공식 가이드라인 방식)
    # 여기서 에러가 난다면 라이브러리 설치가 덜 된 것이므로 Reboot이 필요합니다.
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception:
        # hub.pull이 실패할 경우를 대비한 기본 프롬프트 백업
        from langchain_core.prompts import PromptTemplate
        template = "Answer the following questions as best you can. You have access to the following tools: {tools}\n\nUse the following format:\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought: {agent_scratchpad}"
        prompt = PromptTemplate.from_template(template)
    
    # 에이전트 생성
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
        with st.spinner("정보를 검색하며 답변을 생성 중입니다..."):
            prompt_query = f"사용자의 질문: {user_input}. 관련 진로 도서를 검색하여 추천하고 상담해줘."
            try:
                response = agent_executor.invoke({"input": prompt_query})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("답변 생성 과정에서 오류가 발생했습니다.")
                st.info("API 키 권한이나 할당량을 확인해 주세요.")
