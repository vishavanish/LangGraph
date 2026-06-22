from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Annotated

from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import os , json
from langchain_core.messages import BaseMessage, HumanMessage , AIMessageChunk, ToolMessage
from fastapi.responses import StreamingResponse
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
import dotenv
dotenv.load_dotenv() 
# -------------google model----------------
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY","")
model  = init_chat_model("google_genai:gemini-3-flash-preview")

# ______________ Hugging face model-------------------
# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-72B-Instruct",
#     huggingfacehub_api_token=os.getenv("HUGGING_FACE_ACCESS_TOKEN"),
#     max_new_tokens=512,
#     temperature=0.7,
# )
# model = ChatHuggingFace(llm=llm)


##-------------helper function--------
def extract_text(content):
    """Normalize message_chunk.content into a plain string,
    regardless of which model/provider produced it."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "content" in block and isinstance(block["content"], str):
                    parts.append(block["content"])
        return "".join(parts)
    return ""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["*"],
)

# 1. Define your tools

## google seach tool - 2
# search_tool = TavilySearchResults(max_results=3,include_raw_content=True)
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is rainy."
    

## tool bind with graph
tools = [search_tool, calculator, get_weather]
llm_with_tools = model.bind_tools(tools)


# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Request Schema
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


# Node
def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": messages + [response]
    }

tool_node = ToolNode(tools)

# Graph
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)


# API Route
@app.post("/chat")
async def chat(request: ChatRequest):

    config = {"configurable": {"thread_id": request.thread_id}}

    # result = workflow.invoke(
    #     {"messages": [HumanMessage(content=request.message)]},
    #     config=config
    # )
    
    # return {
    #     "reply": response.content
    # }
    print(f"[ROUTE HIT] message='{request.message}' thread_id='{request.thread_id}'")

    def generate():
        try:
            for chunk in workflow.stream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="messages"
            ):
                message_chunk, metadata = chunk

                if isinstance(message_chunk, AIMessageChunk):
                    # Emit tool call event when model decides to use a tool
                    if message_chunk.tool_call_chunks:
                        for tc in message_chunk.tool_call_chunks:
                            if tc.get("name"):
                                yield f"data: {json.dumps({'tool_call': tc.get('name'), 'args': tc.get('args', '')})}\n\n"

                    # Emit text delta
                    content = extract_text(message_chunk.content)
                    if content:
                        yield f"data: {json.dumps({'delta': content})}\n\n"

                elif isinstance(message_chunk, ToolMessage):
                    # Emit tool result when tool finishes
                    yield f"data: {json.dumps({'tool_result': message_chunk.name, 'output': str(message_chunk.content)})}\n\n"

            print("[GENERATOR DONE]")
            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"[GENERATOR ERROR] {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"   
            
                
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "http://localhost:5173",
        },
    )
    
# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)