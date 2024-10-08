# Main.py
import os
import tempfile
import streamlit as st
from Rag import ChatPDF
from retrying import retry

st.set_page_config(page_title="ChatPDF")

# Initialize session state variables if they are not already initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "assistant" not in st.session_state:
    st.session_state["assistant"] = ChatPDF()
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "thinking_spinner" not in st.session_state:
    st.session_state["thinking_spinner"] = st.empty()
if "ingestion_spinner" not in st.session_state:
    st.session_state["ingestion_spinner"] = st.empty()

# Function to display chat messages
def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        # Remove the key argument to avoid Streamlit warnings
        st.write(f"{'User:' if is_user else 'Assistant:'} {msg}")
    st.session_state["thinking_spinner"] = st.empty()


# Retry logic to process user input with retries in case of errors
@retry(stop_max_attempt_number=3, wait_fixed=2000)  # Retry 3 times, wait 2 seconds between retries
def process_input():
    try:
        if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
            user_text = st.session_state["user_input"].strip()
            with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
                # Get assistant response
                agent_text = st.session_state["assistant"].ask(user_text)

                # Check if the response indicates no relevant documents
                if "no relevant docs" in agent_text.lower():
                    st.warning("No relevant documents were found. Please try a different query.")

            # Append messages to chat history
            st.session_state["messages"].append((user_text, True))
            st.session_state["messages"].append((agent_text, False))
            st.session_state["user_input"] = ""  # Clear the input box after processing
    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise  # Reraise the exception to trigger the retry

# Function to read and process the uploaded file
def read_and_save_file():
    # Clear previous session state and messages
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # Loop through uploaded files and process them
    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file.getbuffer())  # Write file content to temporary file
            file_path = tf.name  # Get the temporary file path

        # Ingest the PDF file into the assistant
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            st.session_state["assistant"].ingest(file_path)  # No tenant argument
        os.remove(file_path)  # Remove the temporary file after processing

# Main page function
def page():
    st.header("ChatPDF")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    display_messages()  # Display chat messages
    st.text_input("Message", key="user_input", on_change=process_input)  # Input for user message

# Run the app
if __name__ == "__main__":
    page()
