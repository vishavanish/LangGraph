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
from langchain_tavily import TavilySearch
from langchain.tools import tool
import asyncio
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


search_tool = TavilySearch(max_results=3,include_raw_content=True)
# search_tool = DuckDuckGoSearchRun(region="us-en")

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



def build_graph():
    # Node
    async def chat_node(state: ChatState):
        messages = state["messages"]

        response = await llm_with_tools.ainvoke(messages)

        return {
            "messages": messages + [response]
        }

    tool_node = ToolNode(tools)

    # Graph
    # conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    # checkpointer = SqliteSaver(conn)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge('tools', 'chat_node')
    graph.add_edge("chat_node", END)
    
    
    chatbot = graph.compile()
    return chatbot

async def main():
    chatbot = build_graph()
    
    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content="defination of LLM in 2 lines?")]},

    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())