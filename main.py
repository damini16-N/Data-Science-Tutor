import streamlit as st
import json
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Streamlit App Configuration
st.set_page_config(page_title="AI-Powered Data Science Guide", layout="wide")
st.title("🤖 AI-Powered Data Science Guide")

# Initialize chat history if not available
if "chat_log" not in st.session_state:
    st.session_state.chat_log = [{"role": "assistant", "content": "Hi there! I’m your AI Data Science Guide. Feel free to ask anything related to Data Science!"}]

#Import Google ChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
f = open('gemini.txt')
GOOGLE_API_KEY = f.read()
#set the openAI key and initilise a chatmodel
chat_model = ChatGoogleGenerativeAI(api_key = GOOGLE_API_KEY, model="gemini-2.0-flash-exp")

# Maintain conversation history
conversation_memory = ConversationBufferMemory(memory_key="chat_records", return_messages=True)

# Define AI Response Template
response_template = PromptTemplate(
    input_variables=["chat_records", "user_query"],
    template="""
    You are an AI expert specializing in Data Science. Respond only to questions related to this field.
    Previous Conversations: {chat_records}
    User Query: {user_query}
    AI Response:"""
)

# Create AI Response Chain
conversation_chain = LLMChain(llm=ai_model, prompt=response_template, memory=conversation_memory)

# Display Previous Chat
for message in st.session_state.chat_log:
    st.chat_message(message["role"]).write(message["content"])

# Accept User Input
user_query = st.chat_input("Enter your Data Science-related question here...")
if user_query:
    st.session_state.chat_log.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    # Generate AI Response
    ai_reply = conversation_chain.run(user_query)
    
    st.session_state.chat_log.append({"role": "assistant", "content": ai_reply})
    st.chat_message("assistant").write(ai_reply)

