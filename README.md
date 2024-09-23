
# ChatPDF

ChatPDF is a Streamlit application that allows users to upload PDF documents and interactively ask questions about the content. It utilizes advanced NLP models to provide relevant answers based on the ingested documents.

## Features

- Upload and ingest PDF documents.
- Chat with an AI assistant based on the content of the PDFs.
- User-friendly interface with real-time interactions.

## Requirements

To run the ChatPDF application, you need the following dependencies. They are listed in `requirements.txt`, but here are the key ones:

- `streamlit`
- `langchain`
- `langchain-community`
- `chatollama`
- `chromadb`
- `huggingface-hub`
- `retrying`
- Other libraries as listed in `requirements.txt`.

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Vignesh02cse/Code-llama.git

   cd Code-llama
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source ./.venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   Install the dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama:**

   To install the Ollama model, follow the instructions specific to your platform:

   - For macOS:

     ```bash
     brew tap ollama/tap
     brew install ollama
     ```

   - For Linux:

     You can use the official installation script:

     ```bash
     curl -sSfL https://ollama.com/download | sh
     ```

   - For Windows:

     Download the installer from the [Ollama website](https://ollama.com/download) and follow the instructions.

5. **Run the application:**

   After installing the dependencies, you can run the Streamlit application with the following command:

   ```bash
   streamlit run Main.py
   ```

6. **Access the application:**

   Open your web browser and go to `http://localhost:8501` to access the ChatPDF application.

## Usage

1. **Upload a PDF document:** Click on the "Upload a document" button and select your PDF file.
2. **Ask questions:** Enter your question in the text input field and click "Send" to receive answers based on the document content.
3. **Chat with the assistant:** Continue the conversation by asking follow-up questions.

