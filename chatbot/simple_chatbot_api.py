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
from langchain_core.messages import BaseMessage, HumanMessage , AIMessageChunk
from fastapi.responses import StreamingResponse
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import dotenv
dotenv.load_dotenv() 
# -------------google model----------------
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY","")
# model  = init_chat_model("google_genai:gemini-3-flash-preview")

# ______________ Hugging face model-------------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_ACCESS_TOKEN"),
    max_new_tokens=512,
    temperature=0.7,
)
model = ChatHuggingFace(llm=llm)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["*"],
)


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

    response = model.invoke(messages)

    return {
        "messages": messages + [response]
    }


# Graph
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
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
                # if chunk["type"] != "messages":
                #     continue

                # message_chunk, metadata = chunk["data"]
                message_chunk, metadata = chunk
                content = message_chunk.content

                # print(f"[CONTENT] {repr(content)}")

                if not content or not isinstance(content, str):
                    continue

                yield f"data: {json.dumps({'delta': content})}\n\n"

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
    
#test
config = {"configurable": {"thread_id": "thread-id-01"}}
result = workflow.invoke(
        {"messages": [HumanMessage(content="HI what is my name")]},
        config=config
    )
print(result)