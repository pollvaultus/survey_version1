import streamlit as st
import os
from langchain_anthropic import ChatAnthropic
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate  # Use langchain_core for PromptTemplate

# Set your actual API key
os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

# Load the GPT model
llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.5)

# Function to load and process the document (with caching)
@st.cache_resource
def load_and_process_documents(file_path):
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    return vectorstore



def main():
    st.title("Survey Analysis Query V2")

    document_paths = [os.path.join(os.getcwd(), "data/document.txt")
                    , os.path.join(os.getcwd(), "data/ceo-selfeval.txt"),  os.path.join(os.getcwd(), "data/consolidated-cleaned.txt") ]

    for document_path in document_paths:
        if not os.path.isfile(document_path):
            st.error(f"Error: '{document_path}' not found. Please make sure the file exists and the path is correct.")
            st.stop()

    # Load and process the documents
    vectorstore = load_and_process_documents(document_paths)

    # Conversation memory for context
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Prompt template for the AI
    template = """You are a helpful assistant designed to analyze survey data and provide information about an organization's structure. 

    You have access to two documents:

    1. **Survey Data:** Contains results and insights from a survey.
    2. **Organizational Chart:** Details the staff hierarchy, titles, and work locations.

    Please answer the user's questions based on the information in these documents. If you can't find the answer, say so clearly.

    Current Conversation:
    {chat_history}

    Human: {question}
    Assistant:"""
    prompt = PromptTemplate(
        input_variables=["chat_history", "question"], template=template
    )

    # Chat interface (with session state for persistence)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant context from FAISS
        docs = vectorstore.similarity_search(prompt, k=3)
        context = "\n".join([x.page_content for x in docs])

        # Conversational QA chain (pass context explicitly)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=vectorstore.as_retriever(), memory=memory
        )
        response = qa_chain({"question": prompt})['answer']  # Don't pass chat_history explicitly here

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()


