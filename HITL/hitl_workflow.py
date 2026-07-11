from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal, Dict, Any
from typing import List, TypedDict, Literal, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage 
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langgraph.types import Command
load_dotenv()
model  = init_chat_model("google_genai:gemini-3-flash-preview")

class State(TypedDict):
    user_input: str
    booking: dict
    human_response: str
    messages: Annotated[list[BaseMessage], add_messages]
    
    
def chat_node(state: State):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def book_flight(state):
    booking = {
        "flight": "AI202",
        "from": "Mumbai",
        "to": "Delhi",
        "price": 6500,
        "status": "Pending"
    }

    return {
        "booking": booking
    }
    

from langchain_core.messages import AIMessage

def confirm_booking(state):
    booking = state["booking"]

    if state["human_response"].lower() == "yes":
        booking["status"] = "Confirmed"
        return {
            "booking": booking,
            "messages": [AIMessage(content="✅ Flight booked successfully.")]
        }

    booking["status"] = "no"
    return {
        "booking": booking,
        "messages": [AIMessage(content="❌ Flight booking cancelled.")]
    }
    
    
def ask_confirmation(state):

    answer = interrupt(
        {
            "message": "Booking created successfully. Confirm?",
            "booking": state["booking"]
        }
    )

    return {
        "human_response": answer
    }
    
def router(state):
    # Find the last HumanMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content.lower()

            if "book" in text and "flight" in text:
                return "book"

            return END

    return END


checkpointer = MemorySaver()
graph = StateGraph(State)

graph.add_node("chat", chat_node)
graph.add_node("book", book_flight)
graph.add_node("ask", ask_confirmation)
graph.add_node("confirm", confirm_booking)

# Start with the chatbot
graph.add_edge(START, "chat")

graph.add_conditional_edges(
    "chat",
    router,
    {
        "book": "book",
        END: END,
    },
)

graph.add_edge("book", "ask")
graph.add_edge("ask", "confirm")
graph.add_edge("confirm", END)


workflow = graph.compile(checkpointer=checkpointer)

# ==helper function
def get_ai_text(message):
    if isinstance(message.content, str):
        return message.content

    if isinstance(message.content, list):
        texts = [
            block["text"]
            for block in message.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(texts)

    return str(message.content)

def main():
    print("Chatbot started. Type 'exit' to quit.\n")

    thread_id = "thread12345"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ")

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = workflow.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        if "__interrupt__" in result:
            interrupt = result["__interrupt__"][0]

            print("\n=== HUMAN APPROVAL REQUIRED ===")
            print(interrupt.value["message"])
            print(interrupt.value["booking"])

            answer = input("\nConfirm (yes/no): ")

            result = workflow.invoke(
                Command(resume=answer),
                config=config,
            )

        # Final output
        if "booking" in result:
            print("\nBooking Status:")
            print(result["booking"])

        if "messages" in result:
            print("AI:", get_ai_text(result["messages"][-1]))

        print("--" * 40)
        
if __name__ == "__main__":
    main()