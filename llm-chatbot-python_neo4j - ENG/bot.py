import streamlit as st
from utils import write_message
from agent import generate_response

# Page Config
st.set_page_config("Finance bot", page_icon=":coin:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วยการเงินของคุณ! มีอะไรให้ช่วยในเรื่องการเงินบ้างไหมครับ?"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler to interact with the agent and return structured outputs.
    """
    try:
        with st.spinner("Processing..."):
            metadata, error = generate_response(message)
            return metadata, error  # Always return two items
    except Exception as e:
        # In case of unexpected error, return empty metadata and error message
        return None, str(e)       

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
    