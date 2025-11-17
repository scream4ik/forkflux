import uuid

import streamlit as st

from app.constants import LLM_AVAILABLE_MODELS, Agent
from app.exceptions import ManualOrchestratorException
from app.orchestrator import ManualOrchestrator
from app.prompts import BRUTALLY_HONEST_PROMPT, CRITIC_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, combine_prompts


@st.cache_resource
def get_orchestrator():
    return ManualOrchestrator()


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agents" not in st.session_state:
        st.session_state.agents = {
            Agent.GENERATOR: {"thread_id": str(uuid.uuid4())},
            Agent.CRITIC: {"thread_id": str(uuid.uuid4())},
        }
    if "is_main_task_set" not in st.session_state:
        st.session_state.is_main_task_set = False
    if "main_task_submitted" not in st.session_state:
        st.session_state.main_task_submitted = False
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = Agent.GENERATOR


initialize_session_state()
orchestrator = get_orchestrator()


def redirect_response() -> None:
    if not st.session_state.messages:
        return

    last_message = st.session_state.messages[-1]
    last_agent_role = last_message["role"]
    last_content = last_message["content"]
    next_agent = Agent.CRITIC if last_agent_role == Agent.GENERATOR else Agent.GENERATOR

    try:
        thread_id = st.session_state.agents[next_agent]["thread_id"]
        response = orchestrator.talk_to(
            agent_name=next_agent, input_text=last_content, thread_id=thread_id, context_from=last_agent_role
        )
        st.session_state.messages.append({"role": next_agent, "content": response})
        st.session_state.current_agent = next_agent
    except ManualOrchestratorException as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Error from {next_agent.value}: {str(e)}"})


with st.sidebar:
    st.title("Configuration")
    agent_generator = st.selectbox(
        "Choose the Generator model",
        LLM_AVAILABLE_MODELS,
        disabled=st.session_state.main_task_submitted,
        help="You can only choose the model before starting the chat.",
    )
    agent_critic = st.selectbox(
        "Choose the Critic model",
        LLM_AVAILABLE_MODELS,
        disabled=st.session_state.main_task_submitted,
        help="You can only choose the model before starting the chat.",
    )
    brutally_honest_mode = st.toggle(
        "Hardcore Mode",
        disabled=st.session_state.main_task_submitted,
        help="The AI will provide direct, unfiltered, and challenging feedback instead of being agreeable.",
    )

    needs_openai_key = agent_generator.startswith("gpt") or agent_critic.startswith("gpt")
    needs_google_key = agent_generator.startswith("gemini") or agent_critic.startswith("gemini")

    openai_api_key = None
    if needs_openai_key:
        openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    google_api_key = None
    if needs_google_key:
        google_api_key = st.text_input("Google API Key", key="google_api_key", type="password")
        "[Get a Google API key](https://ai.google.dev/)"

st.set_page_config(page_title="ForkFlux: AI Collaboration System", page_icon="🤖")
st.title("💬 ForkFlux")
st.caption("🚀 Iteratively improving ideas with collaborating AI agents")

for msg in st.session_state.messages:
    role_str = msg["role"].value if isinstance(msg["role"], Agent) else msg["role"]
    with st.chat_message(role_str):
        st.markdown(msg["content"])

if st.session_state.messages:
    last_message_role = st.session_state.messages[-1]["role"]
    if last_message_role in [Agent.GENERATOR, Agent.CRITIC]:
        next_agent_display = "Critic" if last_message_role == Agent.GENERATOR else "Generator"
        st.button(f"Redirect response to {next_agent_display}", on_click=redirect_response)

if prompt := st.chat_input("What is the main task?"):
    if (needs_openai_key and not openai_api_key) or (needs_google_key and not google_api_key):
        st.info("Please add all required API keys in the sidebar to continue.")
        st.stop()

    st.session_state.main_task_submitted = True

    if not st.session_state.is_main_task_set:
        style_prompt = BRUTALLY_HONEST_PROMPT if brutally_honest_mode else None

        final_generator_prompt = combine_prompts(base_prompt=GENERATOR_SYSTEM_PROMPT, style_prompt=style_prompt)
        final_critic_prompt = combine_prompts(base_prompt=CRITIC_SYSTEM_PROMPT, style_prompt=style_prompt)

        orchestrator.set_llm_api_keys(openai_key=openai_api_key, google_key=google_api_key)
        orchestrator.set_main_task(prompt)
        orchestrator.add_agent(
            name=Agent.GENERATOR,
            system_prompt=final_generator_prompt,
            model=agent_generator,
        )
        orchestrator.add_agent(
            name=Agent.CRITIC,
            system_prompt=final_critic_prompt,
            model=agent_critic,
        )
        st.session_state.is_main_task_set = True

    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        thread_id = st.session_state.agents[st.session_state.current_agent]["thread_id"]
        response = orchestrator.talk_to(
            agent_name=st.session_state.current_agent, input_text=prompt, thread_id=thread_id
        )
        st.session_state.messages.append({"role": st.session_state.current_agent, "content": response})
    except ManualOrchestratorException as e:
        st.session_state.messages.append({"role": "assistant", "content": str(e)})

    st.rerun()
