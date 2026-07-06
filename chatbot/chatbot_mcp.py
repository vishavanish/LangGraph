from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Annotated

from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import os , json, sys
from langchain_core.messages import BaseMessage, HumanMessage , AIMessageChunk, ToolMessage
from fastapi.responses import StreamingResponse
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
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

client = MultiServerMCPClient(
    {
    
        "arithmetic_mcp_server": {
            "transport": "stdio",
            "command": sys.executable,         
            "args": [r"D:\MCP\custom_mcp.py"]
        }
    }
)


#search_tool = TavilySearch(max_results=3,include_raw_content=True)




# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():
    
    tools = await client.get_tools()
    print(tools)
    ## tool bind with graph
    llm_with_tools = model.bind_tools(tools)
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
    chatbot = await build_graph()
    
    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content="Find the multiplication of 10 and 3")]},

    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())