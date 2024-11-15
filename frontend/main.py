import asyncio
import os
from collections.abc import AsyncGenerator
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from client import AgentClient
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData

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

class StreamlitChatApp:
    def __init__(self):
        self.app_title = "Langraph research assistant"
        self.app_icon = "🧰"
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self):
        st.set_page_config(
            page_title=self.app_title,
            page_icon=self.app_icon,
            menu_items={},
        )
        self._hide_streamlit_elements()

    def _hide_streamlit_elements(self):
        st.html(
            """
            <style>
            [data-testid="stStatusWidget"] {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                }
            </style>
            """
        )
        if st.get_option("client.toolbarMode") != "minimal":
            st.set_option("client.toolbarMode", "minimal")
            asyncio.sleep(0.1)
            st.rerun()

    def initialize_session_state(self):
        if "agent_client" not in st.session_state:
            agent_url = os.getenv("AGENT_URL", "http://localhost")
            st.session_state.agent_client = AgentClient(agent_url)

        if "thread_id" not in st.session_state:
            thread_id = st.query_params.get("thread_id")
            if not thread_id:
                thread_id = get_script_run_ctx().session_id
                messages = []
            else:
                history = st.session_state.agent_client.get_history(thread_id=thread_id)
                messages = history.messages
            st.session_state.messages = messages
            st.session_state.thread_id = thread_id

    def render_sidebar(self):
        with st.sidebar:
            st.header(f"{self.app_icon} {self.app_title}")
            st.write("")
            with st.popover("⚙️ Settings", use_container_width=True):
                model_name = st.radio("LLM Model", options=MODELS.keys())
                model = MODELS[model_name]
                st.session_state.agent_client.agent = st.selectbox(
                    "Agent Type",
                    options=AGENT_TYPES,
                )
                use_streaming = st.toggle("Enable Streaming", value=True)
            
            st.markdown(f"Thread ID: **{st.session_state.thread_id}**")

    async def handle_messages(self, messages_agen: AsyncGenerator[ChatMessage | str, None], is_new: bool = False):
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

            if not isinstance(msg, ChatMessage):
                st.error(f"Invalid message type: {type(msg)}")
                st.stop()

            await self._process_message(msg, last_message_type, streaming_placeholder, 
                                     streaming_content, is_new)
            last_message_type = msg.type

    async def _process_message(self, msg, last_message_type, streaming_placeholder, 
                             streaming_content, is_new):
        if msg.type == "human":
            st.chat_message("human").write(msg.content)
        elif msg.type == "ai":
            if is_new:
                st.session_state.messages.append(msg)
            if last_message_type != "ai":
                st.session_state.last_message = st.chat_message("ai")
            await self._handle_ai_message(msg, streaming_placeholder, streaming_content)
        elif msg.type == "custom":
            await self._handle_custom_message(msg, is_new, last_message_type)

    async def _handle_ai_message(self, msg, streaming_placeholder, streaming_content):
        with st.session_state.last_message:
            if msg.content:
                if streaming_placeholder:
                    streaming_placeholder.write(msg.content)
                else:
                    st.write(msg.content)
            
            if msg.tool_calls:
                await self._process_tool_calls(msg)

    async def _process_tool_calls(self, msg):
        call_results = {}
        for tool_call in msg.tool_calls:
            status = st.status(
                f"Tool Call: {tool_call['name']}",
                state="running",
            )
            call_results[tool_call["id"]] = status
            status.write(f"Input: {tool_call['args']}")

        for _ in range(len(call_results)):
            tool_result = await anext(messages_agen)
            if tool_result.type == "tool":
                if is_new:
                    st.session_state.messages.append(tool_result)
                status = call_results[tool_result.tool_call_id]
                status.write(f"Output: {tool_result.content}")
                status.update(state="complete")

    async def handle_feedback(self):
        if "last_feedback" not in st.session_state:
            st.session_state.last_feedback = (None, None)

        latest_run_id = st.session_state.messages[-1].run_id
        feedback = st.feedback("stars", key=latest_run_id)

        if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
            normalized_score = (feedback + 1) / 5.0
            await st.session_state.agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "User feedback"},
            )
            st.session_state.last_feedback = (latest_run_id, feedback)
            st.toast("Feedback recorded", icon="⭐")

    async def run(self):
        self.render_sidebar()
        messages = st.session_state.messages

        if len(messages) == 0:
            with st.chat_message("ai"):
                st.write("Hello! I'm an AI-powered assistant. How can I help you today?")

        async def message_iter():
            for m in messages:
                yield m

        await self.handle_messages(message_iter())

        if user_input := st.chat_input():
            messages.append(ChatMessage(type="human", content=user_input))
            st.chat_message("human").write(user_input)
            
            if st.session_state.get('use_streaming', True):
                stream = st.session_state.agent_client.astream(
                    message=user_input,
                    model=st.session_state.get('model', MODELS["OpenAI GPT-4o-mini"]),
                    thread_id=st.session_state.thread_id,
                )
                await self.handle_messages(stream, is_new=True)
            else:
                response = await st.session_state.agent_client.ainvoke(
                    message=user_input,
                    model=st.session_state.get('model', MODELS["OpenAI GPT-4o-mini"]),
                    thread_id=st.session_state.thread_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()

        if len(messages) > 0:
            with st.session_state.last_message:
                await self.handle_feedback()

if __name__ == "__main__":
    app = StreamlitChatApp()
    asyncio.run(app.run())