import streamlit as st
from streamlit_mic_recorder import mic_recorder
from groq import Groq
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from duckduckgo_search import DDGS
from langchain.tools import tool
import json
from linkup import LinkupClient
from pprint import pp
from datetime import datetime



# --- Constants ---
CONVERSATION_FILE = "conversation_data.json"

# Initialize secrets and Groq client
load_dotenv()
os.environ['GROQ_API_KEY'] = 'GROQ_API_KEY'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it.")
    st.stop()

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not found in environment variables. Please set it.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize session state
def initialize_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'audio_key' not in st.session_state:
        st.session_state.audio_key = {
            "Person1": 0,
            "You": 0
        }
    if 'current_person' not in st.session_state:
        st.session_state.current_person = "Person1"
    if 'new_conversation' not in st.session_state:
        st.session_state.new_conversation = False
    
    # Load conversation history from file if it exists
    if os.path.exists(CONVERSATION_FILE):
      try:
        with open(CONVERSATION_FILE, "r") as f:
            st.session_state.conversation_history = json.load(f)
      except json.JSONDecodeError:
         st.error(f"Error loading conversation data, check the {CONVERSATION_FILE} file.")

def save_conversation_history():
    """Save the conversation history to a JSON file."""
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(st.session_state.conversation_history, f, indent=4)

def transcribe_audio(audio_bytes, filename):
    """Transcribe audio using Groq."""
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
        return transcription.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def display_conversations():
    """Display the conversation history in a dialogue-style format."""
    for item in st.session_state.conversation_history:
      if item.get("helpme_pressed"):
        st.markdown("------------------------------------")
        st.markdown("**Help Me Press Identified:**")
      for message in item.get("conversation", []):
        if message["role"] == "user":
          if message["person"] == "Person1":
            st.markdown(f'<div class="chat-message user-message-left"><b>Person1:</b> {message["content"]}</div>', unsafe_allow_html=True)
          elif message["person"] == "You":
            st.markdown(f'<div class="chat-message user-message-right"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
      if item.get("identified_question"):
        st.markdown(f"Summarized Question to be answered: **{item['identified_question']}**")

      if item.get("answer"):
        st.markdown(f"**Answer:** {item['answer']}")
       
    

def process_voice_input(audio):
    """Process voice input and generate response."""
    try:
        with st.spinner("Transcribing audio..."):
            transcribed_query = transcribe_audio(audio['bytes'], "audio.wav")

            if transcribed_query:
                if st.session_state.new_conversation or not st.session_state.conversation_history:
                    st.session_state.conversation_history.append({"conversation":[], "identified_question": None, "answer": None})
                    st.session_state.new_conversation = False
                
                current_conversation = st.session_state.conversation_history[-1]["conversation"]
                current_conversation.append({
                    "role": "user",
                    "person": st.session_state.current_person,
                    "content": transcribed_query
                })
                st.session_state.conversation_history[-1]["conversation"] = current_conversation
                
                save_conversation_history() # Save after adding new voice input
                st.rerun()

    except Exception as e:
        st.error(f"Error processing voice input: {str(e)}")

def identify_question(transcription):
    """Identifies a question in the given transcription using Groq."""
    try:
        with st.spinner("Identifying question..."):
            # First, try to get the question with a more direct prompt
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Your task is to analyze the last turn of the provided conversation transcript and extract the question that directly implies the user asked. If no question is stated, return the entire content. Return only the question itself (or the entire content if no question is identified) and no additional explanations. The conversation is: \n{transcription}"""
                    }
                ],
                temperature=0.1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            identified_question = completion.choices[0].message.content.strip()
            if identified_question.endswith("?"):
                return identified_question

            # Second try with a simplified approach
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role":"user",
                        "content": f"""Analyze the following transcription and extract the question that was asked, otherwise return the full transcription. The transcription is: {transcription}."""

                    }
                ],
                temperature=0.1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            return completion.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error identifying question: {str(e)}")
        return None

# --- Define Custom Tools ---
@tool("Search the web ")
def web_search(query):
    """
    Searches the web using DuckDuckGo and summarizes the results.
    Args:
        query (str): The search query.
    Returns:
        str: A summarized version of the search results.
    """
    client = LinkupClient(api_key=LINKUP_API_KEY)

    response = client.search(
    query="what is current weather in Mumbai",
    depth="standard",
    output_type="searchResults"
    )
    return f"Todays time:{datetime.now()} "+str(response) 
   

@tool("Query the External knowledge base ")
def knowledge_base_query(query):
    """
    Queries a knowledge base (resume data) using Pinecone for retrieval.
    Args:
        query (str): The query for resume knowledge base.
    Returns:
        str: The content retrieved from the knowledge base.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    assistant = pc.assistant.Assistant(assistant_name="resume")
    msg = Message(content=str(query))
    resp = assistant.chat(messages=[msg])
    return resp["message"]["content"]

    

@tool("Search previous conversations for an answer")
def search_previous_conversations(query):
    """
    Searches the previous conversation history for a relevant answer.
    Args:
        query (str): The query to search for.
    Returns:
        str: A relevant answer from past conversation, or "not found"
    """
    conversation_history = st.session_state.conversation_history
    if not conversation_history:
        return "not found"

    #prepare the prompt for llm
    prompt = f"""Given the user query: "{query}", search through the following previous conversation history, to find if a relevant answer was provided before. If an answer is found, return the answer, otherwise return "not found".

    Previous Conversation History:
    """
    for item in conversation_history:
      for message in item.get("conversation", []):
        prompt += f"\n{message['person']}: {message['content']}"
      if item.get("identified_question"):
         prompt+= f"\nQuestion:{item['identified_question']}"
      if item.get("answer"):
         prompt+= f"\nAnswer:{item['answer']}"
      prompt+="\n---"

    try:
      completion = groq_client.chat.completions.create(
          model="llama-3.1-8b-instant",
          messages=[
                {
                    "role":"user",
                     "content":prompt
                 }
           ],
           temperature=0.1,
           max_tokens=1024,
           top_p=1,
           stream=False,
           stop=None
      )
      answer = completion.choices[0].message.content.strip()
      return str(answer)

    except Exception as e:
        st.error(f"Error during search previous conversations: {str(e)}")
        return "not found"



# --- Define Agents ---
discussion_agent = Agent(
    role='Meeting Participant',
    goal='Contribute to the meeting discussion accurately and concisely, using the most appropriate method for each query.',
    backstory="You are a Helpfull assistant that is a part of a discussion between person1 and you. You have access to various tools to provide accurate and relevant information to the questions asked during the discussion.",
    verbose=True,
    llm=LLM(model="groq/mixtral-8x7b-32768"),
    tools=[search_previous_conversations, web_search, knowledge_base_query],
     instructions="""
    You will be given a question that needs to be answered. You have access to tools to answer the question.
    - You MUST use the "Search previous conversations for an answer" tool with the query first.if no relevant answer found in the previous conversations, you should proceed with the following steps:
    - If the question is a simple and general query that dosen't require any external data, answer it directly.
    - If the question requires a particular information that requires a specific source of External source of data the use query external knowledge base i.e, RAG .
    - If the question can't be answered by Your knowledge Then use web search.
    You should ALWAYS format your final answer as a concise direct answer based on tool output or from your knowledge, if you can directly answer. If you used any tool to get the answer you should mention it.
    """
)

# --- Define Tasks ---
discussion_task = Task(
    description="""ANSWERING PROCESS:

   1. ALWAYS start by using "Search previous conversations for an answer" tool with the provided query.

  2. IF NO RELEVANT PREVIOUS ANSWER EXISTS:

   A. DIRECT KNOWLEDGE RESPONSE
      - Use if you have high confidence in answering without external tools
      - Answer must be based on core knowledge, not speculation
      - Must be factual, verifiable information
      - Should be complete without needing supplementary data
      
   B. EXTERNAL KNOWLEDGE BASE QUERY
      - Use when answer requires:
          • Technical specifications
          • Historical data
          • Procedural information
          • Detailed statistics
          • Company-specific information
          • Product documentation
      - Tool provides curated, verified information
      
   C. WEB SEARCH TOOL
      - Use ONLY for:
          • Current events (last 6 months)
          • Recent market trends
          • New product releases
          • Breaking news
          • Real-time data
      - Must be information that requires up-to-date sources
      - Never use for historical or static information

NOTE: Always choose the most appropriate single tool - do not combine multiple tools for one answer
  RESPONSE FORMAT:
  - Keep answers under 40 words
  - Be direct and concise
  - Only include information from tool results or direct knowledge
  - State which tool was used at the end of the response

   IMPORTANT: Do not add extra information beyond what's required and don't add any extra information Than what the tools have returned.
   
   
   """,
    agent=discussion_agent,
    expected_output="Provide a concise and direct answer to the question, drawing from your knowledge or utilizing a specific tool. Ensure that the name of the tool used is clearly emphasized at the end of your response in the following format: <Tool Used: TOOL_NAME>. Your final answer must be less than 40 words. Make sure to follow both instructions precisely."

)
# --- Create Crew ---
crew = Crew(
    agents=[discussion_agent],
    tasks=[discussion_task],
    verbose=True
)

def run_crewai(user_query):
    """Runs the CrewAI agent with the provided user query."""
    # Set the discussion point
    discussion_task.description = f"""Answer the question: {user_query} using the single most appropriate method and provide the output without adding anything extra.
    ANSWERING PROCESS:

   1. ALWAYS start by using "Search previous conversations for an answer" tool with the provided query.

  2. IF NO RELEVANT PREVIOUS ANSWER EXISTS:

   A. DIRECT KNOWLEDGE RESPONSE
      - Use if you have high confidence in answering without external tools
      - Answer must be based on core knowledge, not speculation
      - Must be factual, verifiable information
      - Should be complete without needing supplementary data
      
   B. EXTERNAL KNOWLEDGE BASE QUERY
      - Use when answer requires:
          • Technical specifications
          • Historical data
          • Procedural information
          • Detailed statistics
          • Company-specific information
          • Product documentation
      - Tool provides curated, verified information
      
   C. WEB SEARCH TOOL
      - Use ONLY for:
          • Current events (last 6 months)
          • Recent market trends
          • New product releases
          • Breaking news
          • Real-time data
          • Recent policy changes
      - Must be information that requires up-to-date sources
      - Never use for historical or static information

NOTE: Always choose the most appropriate single tool - do not combine multiple tools for one answer
  RESPONSE FORMAT:
  - Keep answers under 40 words
  - Be direct and concise
  - Only include information from tool results or direct knowledge
  - State which tool was used at the end of the response

   IMPORTANT: Do not add extra information beyond what's required and don't add any extra information Than what the tools have returned.
   
    """
    result = crew.kickoff()
    return result

def main():
    # Page configuration
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main { padding: 2rem; color: #1a1a1a; }
        .chat-message { margin: 0.5rem 0; padding: 0.5rem; border-radius: 0.5rem; color: #1a1a1a; width: fit-content; margin-bottom: 0.5rem; clear: both; }
        .user-message-left { background-color: #ffffff; float: left; }
        .user-message-right { background-color: #DCF8C6; float: right; }
        .conversation-area { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; max-height: 500px; overflow-y: auto; background-color: #e0e0e0; }
        #the-title { text-align: center; font-family: 'Arial', sans-serif; font-size: 36px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 id="the-title">Assist Me</h1>', unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    

    # Conversation area
    with st.container():
        st.markdown('<div class="conversation-area">', unsafe_allow_html=True)
        display_conversations()
        st.markdown('</div>', unsafe_allow_html=True)

    # Right-aligned buttons and mic recorders
    col1, col2 = st.columns([4, 1])
    with col1:
        # "Start Speaking" button for Person1
        audio1 = mic_recorder(
            start_prompt="Start Speaking (Person1)",
            stop_prompt="Stop Recording",
            just_once=True,
            key=f"voice_recorder_Person1_{st.session_state.audio_key['Person1']}"
        )
    with col2:
        # "Help me" Button
        if st.button("Help me"):
            if st.session_state.conversation_history:
                last_conversation = st.session_state.conversation_history[-1]["conversation"]
                full_transcription = " ".join([msg["content"] for msg in last_conversation]) if last_conversation else ""

                if full_transcription:
                    identified_question = identify_question(full_transcription)

                    if identified_question:
                      answer = run_crewai(identified_question) #get tool used to answer

                      # Ensure that we store only strings in the conversation history.
                      if answer:
                        st.session_state.conversation_history[-1]["identified_question"] = identified_question
                        st.session_state.conversation_history[-1]["answer"] = str(answer) # Convert CrewOutput to str
                        
                      else:
                        st.session_state.conversation_history[-1]["identified_question"] = "No Question identified"
                        st.session_state.conversation_history[-1]["answer"] = "No Answer Provided"
                        

                      # mark that help me was pressed, create a new conversation and mark that it is a new conversation
                      st.session_state.conversation_history[-1]["helpme_pressed"] = True
                      st.session_state.conversation_history.append({"conversation":[], "identified_question": None, "answer": None})
                      st.session_state.new_conversation = True

                    else:
                        st.session_state.conversation_history[-1]["identified_question"] = "No Question identified"
                        st.session_state.conversation_history[-1]["answer"] = None
                        st.session_state.new_conversation = True
                else:
                    st.session_state.conversation_history[-1]["identified_question"] = "No Transcriptions available"
                    st.session_state.conversation_history[-1]["answer"] = None
                    st.session_state.new_conversation = True

                save_conversation_history() #Save conversation history after generating an answer
                st.rerun()
            else:
                st.warning("No conversation available to identify a question.")

        # "Start Speaking" button for You
        audio2 = mic_recorder(
            start_prompt="Start Speaking (You)",
            stop_prompt="Stop Recording",
            just_once=True,
            key=f"voice_recorder_you_{st.session_state.audio_key['You']}"
        )

    # Process voice input if available
    if audio1:
        st.session_state.current_person = "Person1"
        process_voice_input(audio1)
        st.session_state.audio_key["Person1"] += 1
    elif audio2:
        st.session_state.current_person = "You"
        process_voice_input(audio2)
        st.session_state.audio_key["You"] += 1

if __name__ == "__main__":
    main()
