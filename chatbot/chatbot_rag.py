
from __future__ import annotations

import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import TypedDict, Annotated
import requests
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
import dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
dotenv.load_dotenv() 
# -------------google model----------------
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY","")
# model  = init_chat_model("google_genai:gemini-2.0-flash-lite")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ______________ Hugging face model-------------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_ACCESS_TOKEN"),
    max_new_tokens=512,
    temperature=0.7,
    timeout=60,
)
model = ChatHuggingFace(llm=llm)
embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"  #model_name="sentence-transformers/paraphrase-albert-small-v2"
)
        
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["*"],
)


# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass
        
# -------------------
# 3. Tools
# -------------------
# search_tool = DuckDuckGoSearchRun(region="us-en")
search_tool = TavilySearch(max_results=3,include_raw_content=False)

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

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
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

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), thread_id: str = Form(...)):
    file_bytes = await file.read()
    result = ingest_pdf(file_bytes, thread_id, filename=file.filename)
    return {"status": "ok", **result}

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)