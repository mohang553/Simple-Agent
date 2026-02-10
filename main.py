from typing import Annotated, Literal, Optional
import os
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


google_key = os.getenv("GOOGLE_API_KEY")

# ==================== Pydantic Models ====================
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    result: Optional[float] = None
    error: Optional[str] = None


# ==================== Tools ====================
@tool
def BODMA(a: float, b: float) -> float:
    """(a^b) / (a*b)"""
    if a == 0 or b == 0:
        raise ValueError("a and b cannot be zero")
    return (a ** b) / (a * b)


@tool
def CODMA(a: float, b: float) -> float:
    """(a*b) / (a^b)"""
    if a ** b == 0:
        raise ValueError("a^b cannot be zero")
    return (a * b) / (a ** b)


tools = [BODMA, CODMA]

# ==================== State ====================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ==================== LLM ====================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=google_key,
)

llm_with_tools = llm.bind_tools(tools)


# ==================== Agent Node ====================
def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ==================== Tool Node ====================
def tool_node(state: State):
    last_message = state["messages"][-1]
    outputs = []

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        if name == "BODMA":
            result = BODMA.invoke(args)
        elif name == "CODMA":
            result = CODMA.invoke(args)
        else:
            result = f"Unknown tool {name}"

        outputs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": outputs}


# ==================== Routing ====================
def should_continue(state: State) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ==================== Graph ====================
def create_agent_graph():
    workflow = StateGraph(State)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


agent_graph = create_agent_graph()

# ==================== RESULT EXTRACTION ====================
def extract_numerical_result(messages):
    """
    ONLY trust the final AIMessage.
    Let the LLM do the math reasoning.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            try:
                return float(message.content.strip())
            except ValueError:
                return None
    return None


# ==================== FastAPI ====================
app = FastAPI(title="BODMA/CODMA Agent API")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Empty query")

        enhanced_query = (
            request.query
            + "\nReturn ONLY the final numerical result. No explanation."
        )

        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=enhanced_query)]}
        )

        messages = result.get("messages", [])
        final_result = extract_numerical_result(messages)

        if final_result is None:
            return ChatResponse(error="Could not compute result")

        return ChatResponse(result=final_result)

    except Exception as e:
        return ChatResponse(error=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


# ==================== Run ====================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
