
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
from langchain_core.messages import BaseMessage, HumanMessage , AIMessageChunk, ToolMessage, SystemMessage

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
from langchain_core.runnables import RunnableConfig
from langchain_community.vectorstores import FAISS
import requests



dotenv.load_dotenv() 
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
INDEED_ACTOR = "misceres~indeed-scraper"
NAUKRI_ACTOR = "louisdeconinck~naukri-job-scraper"
# # -------------google model----------------
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
    model_name="sentence-transformers/paraphrase-albert-small-v2"  #jdr3"Qwen/Qwen3-Embedding-0.6B"  #model_name="sentence-transformers/paraphrase-albert-small-v2"
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
    return _THREAD_RETRIEVERS.get(str(thread_id))


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



def _run_apify_actor(actor_id: str, payload: dict) -> list[dict]:
    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    resp = requests.post(url, params={"token": APIFY_TOKEN}, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _search_indeed(position: str, location: str, max_items: int) -> list[dict]:
    items = _run_apify_actor(INDEED_ACTOR, {
        "position": position,
        "location": location,
        "maxItems": max_items,
        "country": "IN",
    })
    return [
        {
            "source": "Indeed",
            "title": it.get("positionName"),
            "company": it.get("company"),
            "location": it.get("location"),
            "salary": it.get("salary"),
            "apply_url": it.get("externalApplyLink") or it.get("url"),
        }
        for it in items
    ]


def _search_naukri(position: str, location: str, max_items: int) -> list[dict]:
    items = _run_apify_actor(NAUKRI_ACTOR, {
        "keywords": [position],
        "maxItems": max_items,
    })
    return [
        {
            "source": "Naukri",
            "title": it.get("title") or it.get("jobTitle"),
            "company": it.get("company") or it.get("companyName"),
            "location": it.get("location") or it.get("placeholders", {}).get("location"),
            "salary": it.get("salary"),
            "apply_url": it.get("jobUrl") or it.get("url"),
        }
        for it in items
    ]


@tool
def find_matching_jobs(config: RunnableConfig, max_items: int = 10) -> dict:
    """
    Search Indeed and Naukri for jobs matching the user's uploaded resume.
    Use this when the user asks to find jobs, see openings, or match jobs to their resume.
    """
    thread_id = config["configurable"]["thread_id"]

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No resume indexed for this chat. Upload a resume PDF first."}

    docs = retriever.invoke("job title, core skills, years of experience, tech stack")
    resume_snippet = "\n".join(d.page_content for d in docs)

    prompt = (
        "From this resume text, output ONLY strict JSON like "
        '{"position": "...", "location": "..."}. '
        "Position should be a short job title/role search term.\n\n"
        f"{resume_snippet}"
    )
    result = model.invoke(prompt)
    try:
        terms = json.loads(extract_text(result.content))
    except Exception:
        terms = {"position": "Software Developer", "location": "India"}

    position = terms["position"]
    location = terms.get("location", "India")

    jobs = []
    errors = []

    try:
        jobs += _search_indeed(position, location, max_items)
    except Exception as e:
        errors.append(f"Indeed: {e}")

    try:
        jobs += _search_naukri(position, location, max_items)
    except Exception as e:
        errors.append(f"Naukri: {e}")

    return {"search_terms": terms, "job_count": len(jobs), "jobs": jobs, "errors": errors}


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
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    """
    thread_id = config["configurable"]["thread_id"]
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool,find_matching_jobs]
llm_with_tools = model.bind_tools(tools)

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Request Schema
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


# Node
def chat_node(state: ChatState, config):
    thread_id = config["configurable"]["thread_id"]
    has_doc = str(thread_id) in _THREAD_RETRIEVERS

    sys_msg = SystemMessage(content=(
        "You have a tool called rag_tool that retrieves content from the user's uploaded resume/PDF for this chat thread. "
        + ("A document IS indexed for this thread — call rag_tool whenever the user asks about their background, "
           "company history, experience, skills, or anything that could be in their resume."
           if has_doc else
           "No document is indexed for this thread yet.")
    ))

    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}



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
                    content = message_chunk.content
                    # Tool content may already be a dict/list (from your @tool functions) or a string
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except (json.JSONDecodeError, TypeError):
                            pass  # leave as plain string, e.g. rag_tool error messages

                    yield f"data: {json.dumps({'tool_result': message_chunk.name, 'output': content})}\n\n"
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