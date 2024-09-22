# Rag.py
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import  HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnableLambda

embedding = HuggingFaceEmbeddings()

class ChatPDF:
    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ingest(self, pdf_file_path: str):
        # Load and split the PDF
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if embedding is None or not hasattr(embedding, 'embed_documents'):
            raise ValueError(
                "Embedding instance is not properly initialized or does not have 'embed_documents' method.")

        # Adding tenant information directly in metadata (if needed)
        for chunk in chunks:
            chunk.metadata = {"tenant": "default_tenant"}  # Change as necessary

        # Create the vector store without the tenant argument
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=embedding)
        if self.vector_store is None:
            raise ValueError("Failed to create the vector store.")

        # Set up the retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.2,  # Lower the threshold
            },
        )

        if self.retriever is None:
            raise ValueError("Failed to create the retriever.")

        def combine_context_and_question(inputs):
            return {"context": inputs["context"], "question": inputs["question"]}

        # Wrap the function in a RunnableLambda
        combine_runnable = RunnableLambda(func=combine_context_and_question)

        # Set up the chain
        self.chain = (
                RunnablePassthrough()
                | (RunnableLambda(func=lambda x: print(f"Input to combine: {x}") or combine_context_and_question(x)))
                | (RunnableLambda(func=lambda x: print(f"After combining: {x}") or x))
                | (RunnableLambda(func=lambda x: print(f"Input to prompt: {x}") or self.prompt.invoke(x)))
                | (RunnableLambda(func=lambda x: print(f"Prompt Output: {x}") or x))
                | (RunnableLambda(func=lambda x: print(f"Input to model: {x}") or self.model.invoke(x)))
                | (RunnableLambda(func=lambda x: print(f"Model Output: {x}") or x))
                | (RunnableLambda(func=lambda x: print(f"Final Output: {x}") or StrOutputParser().invoke(x)))
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        # Retrieve relevant documents
        context_docs = self.retriever.get_relevant_documents(query)

        if not context_docs:
            return "No relevant documents found."

        # Extract text from the retrieved documents
        context_texts = "\n".join([doc.page_content for doc in context_docs])

        # Prepare the input for the chain in the expected dictionary format
        inputs = {
            "context": context_texts,
            "question": query
        }

        try:
            response = self.chain.invoke(inputs)
            return response
        except Exception as e:
            return f"An error occurred while processing the input: {e}"

    def clear(self):
        """Reset the internal state of the ChatPDF instance."""
        self.vector_store = None
        self.retriever = None
        self.chain = None
