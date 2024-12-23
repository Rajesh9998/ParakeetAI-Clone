from groq import Groq
import json
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Define models
ROUTING_MODEL = "llama3-70b-8192"  # Or use a smaller model for routing if needed
TOOL_USE_MODEL = "llama3-groq-70b-8192-tool-use-preview"  # RAG use groq model
GENERAL_MODEL = "llama3-70b-8192"  # General LLM use groq model
# Initialize Pinecone assistant
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
assistant = pc.assistant.Assistant(assistant_name= "YOUR_PINECONE_RAG_ASSISTANT_NAME")


# Function to get content via pinecone RAG
def get_rag_response(query):
    msg = Message(content=query)
    try:
        resp = assistant.chat(messages=[msg])
        return resp["message"]["content"]
    except Exception as e:
        return json.dumps({"error": f"Error in RAG: {str(e)}"})

# Function to get online search results
def get_online_search_results(query):
    try:
        content = DDGS().text(query, max_results=5)
        if content:
           result = DDGS().chat(f"Query: {query}. Summarize the information from the following search results in a concise manner, in bullet points, not more than 6-7 Bulleted points: {content}", model='gpt-4o-mini')
           return result
        else :
            return json.dumps({"error": "No search results found."})
    except Exception as e:
        return json.dumps({"error": f"Error in DDGS: {str(e)}"})



def route_query(query):
    """Routing logic to let LLM decide if tools are needed"""
    routing_prompt = f"""
    Given the following user query, determine the best tool to answer it and respond with one of the following:

    - 'TOOL: RAG' if the query is a personal question about the applicant based on their resume or personal experiences for Example : Tell me about your self, what are your previous experiences.
    - 'TOOL: SEARCH' if the query is about current events, real-time information, who is the current/latest information for anything  or trending topics or latest news.
    - 'LLM' if it's a general knowledge question that can be answered with common knowledge and does not need any external source or tools.

    User query: {query}

    Response:
    """

    response = client.chat.completions.create(
        model=ROUTING_MODEL,
        messages=[
            {"role": "system", "content": "You are a routing assistant. Determine the best tool to use based on the user query."},
            {"role": "user", "content": routing_prompt}
        ],
        max_tokens=20
    )
    routing_decision = response.choices[0].message.content.strip()
    if "TOOL: RAG" in routing_decision:
        return "rag tool needed"
    elif "TOOL: SEARCH" in routing_decision:
         return "search tool needed"
    elif "LLM" in routing_decision:
        return "llm"
    else:
        return "llm"



def run_general(query):
    """Use the general model to answer the query since no tool is needed"""
    response = client.chat.completions.create(
        model=GENERAL_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Summarize the information in a concise manner, in bullet points, not more than 50 words"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


def process_query(query):
    """Process the query and route it to the appropriate model"""
    try:
        if not query or not isinstance(query, str):
            return {
                "query": query,
                "route": "error",
                "response": "Invalid query provided"
            }

        print(f"Processing query: {query}")
        
        try:
            route = route_query(query)
        except Exception as e:
            print(f"Error in routing, defaulting to LLM: {str(e)}")
            route = "llm"  # Default to LLM if routing fails
            
        print(f"Route determined: {route}")

        try:
            if route == "rag tool needed":
                response = get_rag_response(query)
                try:
                    response = json.loads(response)
                    response = response['result']
                except:
                    response = response  # Use raw response if parsing fails
            elif route == "search tool needed":
                try:
                    response = get_online_search_results(query)
                    response = json.loads(response)['result']
                except:
                    # Fallback to LLM if search fails
                    print("Search failed, falling back to LLM")
                    response = run_general(query)
            else:
                response = run_general(query)

        except Exception as e:
            print(f"Error in tool execution, falling back to LLM: {str(e)}")
            try:
                response = run_general(query)
            except:
                response = "I apologize, but I'm having trouble generating a response at the moment."

        print(f"Generated response: {response}")

        return {
            "query": query,
            "route": route,
            "response": response
        }
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        return {
            "query": query,
            "route": "error",
            "response": "I apologize, but I'm having trouble processing your request. Please try again."
        }

# Example usage
if __name__ == "__main__":
    queries = [
        "What is your previous work experience?",
        #"Who is the current president of the US?",
         #"What's new in AI today?"
    ]

    for query in queries:
        result = process_query(query)
        print(f"Query: {result['query']}")
        print(f"Route: {result['route']}")
        print(f"Response: {result['response']}\n")