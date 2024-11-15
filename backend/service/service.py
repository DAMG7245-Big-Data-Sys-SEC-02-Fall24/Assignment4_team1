import json
import logging
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from fastapi.openapi.utils import get_openapi
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient
from jose import JWTError
from sqlalchemy.orm import Session
from backend.routes import auth_routes
from backend.services.auth_service import verify_token
from backend.services.database_service import get_db
from backend.agents.agents import DEFAULT_AGENT, agents
from backend.schema.schema import (
    ChatHistory, ChatHistoryInput, ChatMessage,
    Feedback, FeedbackResponse, StreamInput, UserInput,
)
from backend.service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from backend.services.auth_service import verify_token
from backend.services.database_service import get_db

# Setup logging and warnings
warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

# JWT Authentication setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        email = verify_token(token)
        if email is None:
            raise credentials_exception
        return email
    except JWTError:
        raise credentials_exception

def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.")),
    ],
) -> None:
    if http_auth.credentials != os.getenv("AUTH_SECRET"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

bearer_depend = [Depends(verify_bearer)] if os.getenv("AUTH_SECRET") else None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        for a in agents.values():
            a.checkpointer = saver
        yield

# Initialize FastAPI app
app = FastAPI(
    title="User Authentication API",
    description="This is a user authentication API with JWT-based protection.",
    version="1.0",
    lifespan=lifespan
)


app.include_router(auth_routes.router, prefix="/auth", tags=["Authentication"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred.", "details": str(exc)}
    )

router = APIRouter(dependencies=bearer_depend)

def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    kwargs = {
        "input": {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model}, run_id=run_id
        ),
    }
    return kwargs, run_id

async def ainvoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    agent: CompiledStateGraph = agents[agent_id]
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = langchain_to_chat_message(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

# Root endpoint
@app.get("/")
def read_root(current_user: str = Depends(get_current_user)):
    return {"message": "Welcome to the User Authentication API"}

# Health check route
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Protected route example
@app.get("/protected")
async def protected_route(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return {"message": f"You have access to this protected route, {current_user}"}

# Route handlers
@router.post("/invoke")
async def invoke(user_input: UserInput, current_user: str = Depends(get_current_user)) -> ChatMessage:
    return await ainvoke(user_input=user_input)

@router.post("/{agent_id}/invoke")
async def agent_invoke(
    user_input: UserInput, 
    agent_id: str, 
    current_user: str = Depends(get_current_user)
) -> ChatMessage:
    return await ainvoke(user_input=user_input, agent_id=agent_id)

async def message_generator(
    user_input: StreamInput, 
    agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    agent: CompiledStateGraph = agents[agent_id]
    kwargs, run_id = _parse_input(user_input)

    async for event in agent.astream_events(**kwargs, version="v2"):
        if not event:
            continue

        new_messages = []
        if (
            event["event"] == "on_chain_end"
            and any(t.startswith("graph:step:") for t in event.get("tags", []))
            and "messages" in event["data"]["output"]
        ):
            new_messages = event["data"]["output"]["messages"]

        if event["event"] == "on_custom_event" and "custom_data_dispatch" in event.get("tags", []):
            new_messages = [event["data"]]

        for message in new_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                continue
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        if (
            event["event"] == "on_chat_model_stream"
            and user_input.stream_tokens
            and "llama_guard" not in event.get("tags", [])
        ):
            content = remove_tool_calls(event["data"]["chunk"].content)
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
            continue

    yield "data: [DONE]\n\n"

def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }

@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(
    user_input: StreamInput,
    current_user: str = Depends(get_current_user)
) -> StreamingResponse:
    return StreamingResponse(message_generator(user_input), media_type="text/event-stream")

@router.post(
    "/{agent_id}/stream", 
    response_class=StreamingResponse, 
    responses=_sse_response_example()
)
async def agent_stream(
    user_input: StreamInput,
    agent_id: str,
    # current_user: str = Depends(get_current_user)
) -> StreamingResponse:
    return StreamingResponse(
        message_generator(user_input, agent_id=agent_id), 
        media_type="text/event-stream"
    )

# @router.post("/feedback")
# async def feedback(
#     feedback: Feedback,
#     current_user: str = Depends(get_current_user)
# ) -> FeedbackResponse:
#     client = LangsmithClient()
#     kwargs = feedback.kwargs or {}
#     client.create_feedback(
#         run_id=feedback.run_id,
#         key=feedback.key,
#         score=feedback.score,
#         **kwargs,
#     )
#     return FeedbackResponse()

@router.post("/history")
async def history(
    input: ChatHistoryInput,
    current_user: str = Depends(get_current_user)
) -> ChatHistory:
    agent: CompiledStateGraph = agents[DEFAULT_AGENT]
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

# Custom OpenAPI Schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title="User Authentication API",
        version="1.0",
        description="This is a user authentication API with JWT-based protection.",
        routes=app.routes,
    )
    
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if "security" in openapi_schema["paths"][path][method]:
                openapi_schema["paths"][path][method]["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Include main router
app.include_router(router)