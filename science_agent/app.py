import os

import streamlit as st
from agents import Agent, FileSearchTool, Runner, SQLiteSession


# ---------------------------------------------------------
# 1. 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Science Inquiry Coach - RAG",
    page_icon="🔬",
)

vector_store_id = os.getenv("VECTOR_STORE_ID")

if not vector_store_id:
    st.error(
        "VECTOR_STORE_ID가 설정되어 있지 않습니다.\n\n"
        "먼저 setup_knowledge.py를 실행한 뒤 출력된 vs_... 값을 "
        "Codespaces Secret 또는 환경변수 VECTOR_STORE_ID로 설정하세요."
    )
    st.stop()


# ---------------------------------------------------------
# 2. Agent Instructions
# ---------------------------------------------------------
INSTRUCTIONS = """
[ROLE]
너는 중학생의 과학 자유탐구 질문 형성을 지원하는
Science Inquiry Coach이다.

[GOAL]
학생이 자신의 관심사를 과학 개념과 연결하고,
탐구 가능한 질문을 스스로 발전시키도록 지원한다.

[KNOWLEDGE USE]
1. 과학 탐구 질문의 기준, 교육과정 연계 내용, 탐구 방법을 설명할 때는
   연결된 Knowledge를 우선 검색한다.
2. Knowledge에서 확인한 내용을 바탕으로 답한다.
3. 자료에서 확인되지 않는 내용은 자료에 있는 것처럼 말하지 않는다.
4. Knowledge에 충분한 근거가 없으면
   "현재 제공된 자료에서는 확인하기 어렵습니다."라고 알린다.
5. 학생의 상황에 따라 추가 정보가 필요하면 먼저 질문한다.

[RULES]
1. 학생의 생각을 먼저 확인한다.
2. 완성된 탐구 질문을 처음부터 바로 제공하지 않는다.
3. 정보가 부족하면 학년, 관심 분야, 초기 아이디어 등을 추가로 질문한다.
4. 질문을 개선할 때는 개선 이유를 함께 설명한다.
5. 여러 방향을 제안하되 최종 선택은 학생이 하도록 한다.
6. 수행평가, 탐구보고서, 과제를 학생 대신 완성하지 않는다.
7. 위험한 실험을 권하지 않는다.
8. 안전성이 불확실하면 교사의 검토를 요청한다.

[PROCESS]
1. 학생의 학년과 관심 분야를 확인한다.
2. 학생의 초기 탐구 아이디어를 확인한다.
3. 필요한 경우 Knowledge를 검색한다.
4. 질문의 명확성, 탐구 가능성, 범위, 측정 가능성, 안전성을 분석한다.
5. 개선이 필요한 이유를 설명한다.
6. 2~3개의 개선 방향을 제안한다.
7. 학생이 하나를 선택하거나 자신의 표현으로 수정하도록 한다.
8. 수정된 질문을 다시 평가한다.

[OUTPUT]
상황에 따라 다음 항목을 사용한다.

- 현재 아이디어에서 좋은 점
- 보완이 필요한 점
- Knowledge에 근거한 설명
- 개선 방향 2~3개
- 학생이 다음에 생각할 질문

답변은 중학생이 이해할 수 있는 표현으로 작성한다.
"""


# ---------------------------------------------------------
# 3. RAG가 연결된 Single Education Agent
# ---------------------------------------------------------
science_agent = Agent(
    name="Science Inquiry Coach",
    instructions=INSTRUCTIONS,
    tools=[
        FileSearchTool(
            vector_store_ids=[vector_store_id],
            max_num_results=3,
        )
    ],
)


# ---------------------------------------------------------
# 4. Streamlit UI History
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# 5. Agent Conversation Context
#    UI History와 Agent Context는 서로 다른 역할을 한다.
# ---------------------------------------------------------
if "agent_session" not in st.session_state:
    st.session_state.agent_session = SQLiteSession(
        "student_session",
        "science_agent.db",
    )


# ---------------------------------------------------------
# 6. 화면 구성
# ---------------------------------------------------------
st.title("Science Inquiry Coach")
st.caption(
    "RAG 기반 Knowledge Education Agent · "
    "교육자료를 검색하여 학생의 과학 탐구질문 형성을 지원합니다."
)

with st.sidebar:
    st.subheader("Week 6 RAG")
    st.write("Knowledge")
    st.code(vector_store_id, language=None)

    st.markdown(
        """
        **테스트 예시**
        - 좋은 과학탐구 질문의 기준은 무엇인가요?
        - 식물 성장 주제를 탐구 질문으로 만들고 싶어요.
        - 미세플라스틱은 왜 문제일까?라는 질문은 좋은가요?
        - 위험한 실험 아이디어가 나오면 어떻게 해야 하나요?
        """
    )


# 이전 UI 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# 7. 사용자 입력 → Agent 실행
# ---------------------------------------------------------
user_input = st.chat_input(
    "과학 탐구에 관심 있는 주제나 질문을 입력하세요."
)

if user_input:
    # UI History에 사용자 메시지 저장
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent 실행
    with st.chat_message("assistant"):
        with st.spinner("Knowledge를 확인하고 있습니다..."):
            try:
                result = Runner.run_sync(
                    science_agent,
                    user_input,
                    session=st.session_state.agent_session,
                )

                response = result.final_output
                st.markdown(response)

            except Exception as e:
                response = f"오류가 발생했습니다: {e}"
                st.error(response)

    # UI History에 Agent 응답 저장
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
