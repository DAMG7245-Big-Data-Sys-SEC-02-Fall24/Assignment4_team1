# app.py
import streamlit as st
from services.session_store import session_store
from services.authentication import auth
import logging
import os
from dotenv import load_dotenv
import asyncio
from client.client import AgentClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Models and Agent Types
MODELS = {
    "OpenAI GPT-4o-mini": "gpt-4o-mini",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Claude 3 Haiku": "claude-3-haiku",
    "llama-3.1-70b": "llama-3.1-70b",
    "AWS Bedrock Haiku": "bedrock-haiku",
}

AGENT_TYPES = [
    "research-assistant",
    "chatbot",
    "bg-task-agent",
]

# Session state defaults
session_defaults = {
    'display_login': True,
    'display_register': False,
    'current_page': 'Chat',
    'thread_id': None,
    'messages': []
}

def initialize_session_state():
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    # Initialize agent client if not exists
    if "agent_client" not in st.session_state:
        agent_url = os.getenv("AGENT_URL", "http://localhost:8000")
        st.session_state.agent_client = AgentClient(agent_url)

def clear_session_storage():
    logging.info("Clearing session storage")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

def display_login_form():
    st.subheader("Login")
    
    with st.form("login_form", clear_on_submit=True):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if not email or not password:
                st.error("Please enter both email and password.")
                return
                
            try:
                auth.login(email, password)
                st.success("Logged in successfully!")
                st.session_state['current_page'] = 'Chat'  # Set to Chat after login
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {str(e)}")

    if st.button("Register"):
        show_register_form()

def display_register_form():
    st.subheader("Register")
    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Register")

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

    if st.button("Back to Login"):
        show_login_form()

def show_register_form():
    session_store.set_value('display_login', False)
    session_store.set_value('display_register', True)
    st.rerun()

def show_login_form():
    session_store.set_value('display_login', True)
    session_store.set_value('display_register', False)
    st.rerun()

def login_page():
    st.markdown(
        """
        <style>
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
        .features h3 {
            color: #1A5276;
        }
        .features ul {
            text-align: center;
            list-style-type: none;
            padding-left: 0;
        }
        .features ul li {
            font-size: 1.1em;
            color: #333;
            margin-bottom: 10px;
            padding-left: 1.5em;
            text-indent: -1.5em;
        }
        .features ul li:before {
            content: '✓';
            margin-right: 10px;
            color: #1ABC9C;
        }
        </style>
        <div class="container">
            <div class="title">AI Chat Assistant</div>            
            <div class="features">
                <h3>Features available:</h3>
                <ul>
                    <li>Multiple AI Models Support</li>
                    <li>Real-time Chat Interface</li>
                    <li>Different Agent Types</li>
                    <li>Message Streaming</li>
                </ul>
            </div>            
        </div>
        """,
        unsafe_allow_html=True
    )

    if session_store.get_value('display_login'):
        display_login_form()
    elif session_store.get_value('display_register'):
        display_register_form()

async def handle_messages(messages_agen, is_new=False):
    last_message_type = None
    st.session_state.last_message = None
    streaming_content = ""
    streaming_placeholder = None

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

        if msg.type == "human":
            st.chat_message("human").write(msg.content)
        elif msg.type == "ai":
            if is_new:
                st.session_state.messages.append(msg)
            if last_message_type != "ai":
                st.session_state.last_message = st.chat_message("ai")
            with st.session_state.last_message:
                if msg.content:
                    if streaming_placeholder:
                        streaming_placeholder.write(msg.content)
                    else:
                        st.write(msg.content)
        
        last_message_type = msg.type

async def chat_page():
    st.title("AI Chat Assistant")

    # Sidebar configuration
    with st.sidebar:
        with st.expander("⚙️ Chat Settings"):
            model_name = st.radio("LLM Model", options=MODELS.keys())
            st.session_state.model = MODELS[model_name]
            
            st.session_state.agent_type = st.selectbox(
                "Agent Type",
                options=AGENT_TYPES,
            )
            
            st.session_state.use_streaming = st.toggle("Enable Streaming", value=True)
        
        st.markdown("---")
        st.markdown(f"Current User: **{st.session_state.get('user_email', 'Not logged in')}**")
        
    # Initialize or get thread_id
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = st.session_state.get('user_email', '')
        try:
            history = st.session_state.agent_client.get_history(thread_id=st.session_state.thread_id)
            st.session_state.messages = history.messages
        except Exception:
            st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        if message.type == "human":
            st.chat_message("human").write(message.content)
        elif message.type == "ai":
            with st.chat_message("ai"):
                st.write(message.content)

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"type": "human", "content": prompt})
        st.chat_message("human").write(prompt)

        if st.session_state.get('use_streaming', True):
            stream = st.session_state.agent_client.astream(
                message=prompt,
                model=st.session_state.get('model', MODELS["OpenAI GPT-4o-mini"]),
                thread_id=st.session_state.thread_id,
            )
            await handle_messages(stream, is_new=True)
        else:
            response = await st.session_state.agent_client.ainvoke(
                message=prompt,
                model=st.session_state.get('model', MODELS["OpenAI GPT-4o-mini"]),
                thread_id=st.session_state.thread_id,
            )
            st.session_state.messages.append(response)
            st.chat_message("ai").write(response.content)

def main():
    initialize_session_state()

    if not session_store.is_authenticated():
        login_page()
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Set Chat as the only option after login
    st.session_state['current_page'] = 'Chat'
    asyncio.run(chat_page())

    # Logout button in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        try:
            logging.info("Logging out user")
            auth.logout()
            clear_session_storage()
            st.success("Logged out successfully.")
            st.rerun()
        except Exception as e:
            logging.error(f"Error during logout: {e}")
            st.sidebar.error("Error during logout. Please try again.")

if __name__ == "__main__":
    main()