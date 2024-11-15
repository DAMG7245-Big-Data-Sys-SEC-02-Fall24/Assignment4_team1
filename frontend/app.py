import asyncio
import os
import logging
from collections.abc import AsyncGenerator
from dotenv import load_dotenv

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData
from services.session_store import session_store
from services.authentication import auth
from app_pages.home_page import home_page

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Constants
APP_TITLE = "AI Chat Platform"
APP_ICON = "💬"
API_BASE_URL = os.getenv("API_BASE_URL")

# Available LLM models
MODELS = {
    "OpenAI GPT-4o-mini (streaming)": "gpt-4o-mini",
    # "Gemini 1.5 Flash (streaming)": "gemini-1.5-flash",
    # "Claude 3 Haiku (streaming)": "claude-3-haiku",
    # "llama-3.1-70b on Groq": "llama-3.1-70b",
    # "AWS Bedrock Haiku (streaming)": "bedrock-haiku",
}

# Session state defaults
SESSION_DEFAULTS = {
    'display_login': True,
    'display_register': False,
    'current_page': 'Home',
    'thread_id': None,
    'messages': [],
    # 'last_feedback': (None, None)
}

def initialize_session_state():
    """Initialize session state with default values."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

def clear_session_storage():
    """Clear all session storage and reinitialize defaults."""
    logging.info("Clearing session storage")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

def setup_page_config():
    """Configure page settings and styling."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Apply custom styling using markdown
    st.markdown("""
        <style>
        [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }
        .container {
            text-align: center;            
            font-family: Arial, sans-serif;
        }
        .title {
            font-size: 2.5em;
            color: #2E86C1;
            margin-bottom: 10px;
        }
        .features {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stButton > button {
            width: 100%;
        }
        .stTextInput > div > div > input {
            background-color: white;
        }
        </style>
    """, unsafe_allow_html=True)

def display_login_page():
    """Display the login page with features overview."""
    st.markdown("""
        <div class="container">
            <div class="title">AI Chat Platform</div>            
            <div class="features">
                <h3>Features available:</h3>
                <ul>
                    <li>Advanced AI Chat Capabilities</li>
                    <li>Multiple Language Model Options</li>
                    <li>Real-time Streaming Responses</li>
                    <li>Interactive Conversation History</li>
                </ul>
            </div>            
        </div>
    """, unsafe_allow_html=True)

    if session_store.get_value('display_login'):
        display_login_form()
    elif session_store.get_value('display_register'):
        display_register_form()

def display_login_form():
    """Display the login form."""
    st.subheader("Login")
    
    with st.form("login_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Please enter both email and password.")
                return
                
            try:
                auth.login(email, password)
                st.success("Logged in successfully!")
                st.session_state['current_page'] = 'Home'
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {str(e)}")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Create New Account", use_container_width=True):
            show_register_form()

def display_register_form():
    """Display the registration form."""
    st.subheader("Create New Account")
    
    with st.form("register_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Register", use_container_width=True)

        if submit:
            if not username or not email or not password:
                st.error("Please fill in all fields.")
                return
                
            try:
                auth.register(username, email, password)
                st.success("Registered successfully! Please log in.")
                show_login_form()
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Back to Login", use_container_width=True):
            show_login_form()

def show_register_form():
    """Switch to registration form view."""
    session_store.set_value('display_login', False)
    session_store.set_value('display_register', True)
    st.rerun()

def show_login_form():
    """Switch to login form view."""
    session_store.set_value('display_login', True)
    session_store.set_value('display_register', False)
    st.rerun()

async def setup_agent_client():
    """Initialize the agent client if not already present."""
    if "agent_client" not in st.session_state:
        agent_url = os.getenv("AGENT_URL", "http://localhost:8000")
        print("-----Acess token---")
        print(st.session_state.get("access_token"))
        st.session_state.agent_client = AgentClient(agent_url)

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            history: ChatHistory = st.session_state.agent_client.get_history(thread_id=thread_id)
            messages = history.messages
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

def setup_sidebar():
    """Configure and display the sidebar."""
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        
        # Agent settings
        with st.expander("Agent Settings"):
            m = st.radio("LLM to use", options=MODELS.keys())
            st.session_state['model'] = MODELS[m]
            st.session_state.agent_client.agent = st.selectbox(
                "Agent to use",
                options=["research-assistant", "chatbot"],
            )
            st.session_state['use_streaming'] = True

        # Navigation
        st.subheader("Navigation")
        pages = {
            "Home": home_page,
            "Chat": None  # Handled separately in main
        }

        current_page = st.session_state['current_page']
        selected_page = st.radio("Go to", list(pages.keys()))

        if selected_page != current_page:
            st.session_state['current_page'] = selected_page
            st.rerun()

        # Logout button
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            try:
                logging.info("Logging out user")
                auth.logout()
                clear_session_storage()
                st.success("Logged out successfully.")
                st.rerun()
            except Exception as e:
                logging.error(f"Error during logout: {e}")
                st.sidebar.error("Error during logout. Please try again.")

async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draw chat messages, handling both existing and streaming messages.
    
    Args:
        messages_agen: Async generator of messages to draw
        is_new: Whether these are new messages being streamed
    """
    last_message_type = None
    st.session_state.last_message = None
    streaming_content = ""
    streaming_placeholder = None

    print("Drawing messages")
    print(messages_agen)

    async for msg in messages_agen:
        if isinstance(msg, str):
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue

        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        if msg.type == "human":
            last_message_type = "human"
            st.chat_message("human").write(msg.content)
        
        elif msg.type == "ai":
            if is_new:
                st.session_state.messages.append(msg)

            if last_message_type != "ai":
                last_message_type = "ai"
                st.session_state.last_message = st.chat_message("ai")

            with st.session_state.last_message:
                if msg.content:
                    if streaming_placeholder:
                        streaming_placeholder.write(msg.content)
                        streaming_content = ""
                        streaming_placeholder = None
                    else:
                        st.write(msg.content)

                if msg.tool_calls:
                    await handle_tool_calls(msg.tool_calls, messages_agen, is_new)

async def handle_tool_calls(tool_calls, messages_agen, is_new):
    """Handle tool calls and their results."""
    call_results = {}
    for tool_call in tool_calls:
        status = st.status(
            f"""Tool Call: {tool_call["name"]}""",
            state="running" if is_new else "complete",
        )
        call_results[tool_call["id"]] = status
        status.write("Input:")
        status.write(tool_call["args"])

    for _ in range(len(call_results)):
        print("Waiting for tool result")
        tool_result: ChatMessage = await anext(messages_agen)
        if tool_result.type != "tool":
            st.error(f"Unexpected ChatMessage type: {tool_result.type}")
            st.write(tool_result)
            st.stop()

        if is_new:
            st.session_state.messages.append(tool_result)
        status = call_results[tool_result.tool_call_id]
        status.write("Output:")
        status.write(tool_result.content)
        status.update(state="complete")

# async def handle_feedback() -> None:
#     """Handle user feedback for messages."""
#     latest_run_id = st.session_state.messages[-1].run_id
#     feedback = st.feedback("stars", key=latest_run_id)

#     if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
#         normalized_score = (feedback + 1) / 5.0
#         await st.session_state.agent_client.acreate_feedback(
#             run_id=latest_run_id,
#             key="human-feedback-stars",
#             score=normalized_score,
#             kwargs={"comment": "In-line human feedback"},
#         )
#         st.session_state.last_feedback = (latest_run_id, feedback)
#         st.toast("Feedback recorded", icon="⭐")

async def chat_interface():
    """Display and handle the chat interface."""
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        with st.chat_message("ai"):
            st.write("Hello! I'm an AI assistant. How can I help you today?")

    async def amessage_iter():
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        
        use_streaming = st.session_state.get('use_streaming', True)
        model = st.session_state.get('model', MODELS["OpenAI GPT-4o-mini (streaming)"])
        
        if use_streaming:
            stream = st.session_state.agent_client.astream(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
                
            )
            print("Streaming response")
            print(stream)
            await draw_messages(stream, is_new=True)
        else:
            response = await st.session_state.agent_client.ainvoke(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
            )
            messages.append(response)
            st.chat_message("ai").write(response.content)
        st.rerun()

    # if len(messages) > 0:
    #     with st.session_state.last_message:
    #         # await handle_feedback()

async def main():
    """Main application entry point."""
    initialize_session_state()
    setup_page_config()
    
    if not session_store.is_authenticated():
        display_login_page()
        return

    await setup_agent_client()
    setup_sidebar()

    current_page = st.session_state['current_page']
    
    if current_page == "Chat":
        await chat_interface()
    elif current_page == "Home":
        home_page()

if __name__ == "__main__":
    asyncio.run(main())

