from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv

# Load .env file
_ = load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Page Config
st.set_page_config(
    page_title='Custom Assistant',
    page_icon='🤖'
)
st.subheader('Custom ChatGPT 🤖')

# LLM
chat = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.5)

# System and Human Messages from user
system_message = st.text_input(label='System Role')
user_prompt = st.text_input('Send a Message')

# Creating the "messages" which is the "chat_histroy" in stremalir session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    # For System Message
    if system_message:
        # if not exits
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
            # I will append it
            st.session_state.messages.append(SystemMessage(content=system_message))

    # For User Messages
    if user_prompt:
        st.session_state.messages.append(HumanMessage(content=user_prompt))

        # Spinner
        with st.spinner('Working on your Request ...'):
            # Creating the GPT response
            response = chat(st.session_state.messages)

        # Adding the response content to the session state
        st.session_state.messages.append(AIMessage(content=response.content))


# Adding a default system Message if the user did not enter one
if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], SystemMessage):   # The first message is not the system message
        # Append it firstly
        st.session_state.messages.insert(0, SystemMessage('You are a helpful assistant.'))

# Displaying the message (chat_history)
for i, msg in enumerate(st.session_state.messages[1:]):

    if i % 2 == 0:
        # From streamlit_chat
        message(msg.content, is_user=True, key=f'{i} + 🤓') # User Question
    
    else:
        # From stramlit_chat
        message(msg.content, is_user=False, key=f'{i} + 🤖') # ChatGPT Response 
