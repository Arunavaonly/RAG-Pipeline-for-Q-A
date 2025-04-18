import streamlit as st
import os
from langchain_groq import ChatGroq
# Import LCEL components
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Your original NCERT instructions, with a {context} slot:
system_template = """
You are a Mathematics and Science teacher for very young children.
Explain concepts in simple terms, with examples and analogies.
Answer only as an NCERT style teacher. When you need to solve an equation, show and explain all mathematical steps clearly.

**CRITICAL FORMATTING RULE:**
* For **ALL** mathematical content (variables, symbols, formulas, equations):
    * Use **ONLY** standard LaTeX delimiters.
    * For inline math, use single dollar signs: `$ ... $`. Example: The variable is $x$. The formula is $E=mc^2$.
    * For display math (equations on their own line), use double dollar signs: `$$ ... $$`. Example: $$ \sum_{{i=1}}^{{n}} i = \frac{{n(n+1)}}{{2}} $$ 
* **DO NOT** use HTML tags like `<span ...>` or any other formatting for mathematics. Only use `$` and `$$`.


Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and directly related to the provided context if possible.

CONTEXT:
{context}
"""

human_template = "{input}"

# Compose into one ChatPromptTemplate:
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template(human_template),
])


load_dotenv()
groq_api_key = os.getenv("Groq_API_Key")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure Google API key is set (handle potential None value)
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Ensure Groq API key is set (handle potential None value)
if not groq_api_key:
    st.error("Groq_API_Key not found in environment variables.")
    st.stop()


# ————————————————————————————————————————————————————————
# Streamlit page setup & CSS (unchanged)
st.set_page_config(page_title="NCERT Tutor", layout="wide") # Added layout="wide" for better column spacing
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
    color: #333333;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 4px;
    cursor: pointer;
}
.stButton > button:hover {
    background-color: #45a049;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#29323c, #485563);
    color: white;
}

/* Added CSS for the scrollable container */
.stContainer {
    overflow-y: auto !important; /* Force scrollbar */
}

</style>
""", unsafe_allow_html=True)

st.title("NCERT Math and Science Tutor")
st.subheader("Ask Questions From NCERT Math and Science Textbooks")

# ————————————————————————————————————————————————————————
# LLMs (Check for API key existence before initializing)
try:
    llm_math = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    llm_science = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
except Exception as e:
    st.error(f"Error initializing LLMs. Check API keys and model availability: {e}")
    st.stop()


# ————————————————————————————————————————————————————————
# (Re‑usable) loader + FAISS builder
def build_faiss(path: str):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        # Check if directory exists and has PDFs
        if not os.path.isdir(path) or not any(fname.endswith('.pdf') for fname in os.listdir(path)):
            st.warning(f"Directory '{path}' not found or contains no PDF files.")
            return None
        docs = PyPDFDirectoryLoader(path).load()
        if not docs:
            st.warning(f"No documents loaded from '{path}'.")
            return None

        text_splitter = SemanticChunker(embeddings, min_chunk_size=100)

        # Handle potential variations in document structure if load() returns different types
        texts = [doc.page_content for doc in docs if hasattr(doc, 'page_content')]
        metadata = [doc.metadata for doc in docs if hasattr(doc, 'metadata')]
        chunks = text_splitter.create_documents(texts, metadatas=metadata) # Pass metadatas if available
        if not chunks:
            st.warning(f"Text splitting resulted in no chunks for '{path}'.")
            return None
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"Error building FAISS index for '{path}': {e}")
        return None

# ————————————————————————————————————————————————————————
# Sidebar buttons to warm up embeddings & vectors
st.sidebar.title("Load Data")
if st.sidebar.button("Load Math Data"):
    with st.spinner("Building Math Vector Store... Please wait."):
        st.session_state.vectors_math = build_faiss("./Mathematics")
        if st.session_state.get("vectors_math"): # Check if successfully created
            st.sidebar.success("Mathematics DB Is Ready")
        else:
            st.sidebar.error("Failed to build Math DB.")


if st.sidebar.button("Load Science Data"):
    with st.spinner("Building Science Vector Store... Please wait."):
        st.session_state.vectors_science = build_faiss("./Science")
        if st.session_state.get("vectors_science"): # Check if successfully created
            st.sidebar.success("Science DB Is Ready")
        else:
            st.sidebar.error("Failed to build Science DB.")

# ————————————————————————————————————————————————————————

# Initialize session state for chat messages if they don't exist
if "messages_math" not in st.session_state:
    st.session_state.messages_math = []
if "messages_science" not in st.session_state:
    st.session_state.messages_science = []

# Initialize memory in session_state once
if "mem_math" not in st.session_state:
    st.session_state.mem_math = ConversationBufferMemory(
        k = 20,
        memory_key="chat_history",
        input_key="input",
        output_key="answer",
        return_messages=True
    )
if "mem_science" not in st.session_state:
    st.session_state.mem_science = ConversationBufferMemory(
        k = 20,
        memory_key="chat_history",
        input_key="input",
        output_key="answer",
        return_messages=True
    )

# ————————————————————————————————————————————————————————
# Removed the unused display_chat_history function

# ————————————————————————————————————————————————————————
# Layout: two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### NCERT Mathematics")
    if "vectors_math" not in st.session_state or st.session_state.vectors_math is None:
        st.warning("Please load Math data first using the sidebar button.")
    else:
        # Create a container for the chat history with a fixed height and scroll
        # Adjust the height (e.g., 400px) as needed based on your layout preference
        with st.container(height=500): # <--- Added container here
            # Display existing chat messages from session state INSIDE the container
            for message in st.session_state.messages_math:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"]) # Use markdown to render potential LaTeX

        # Accept user input using st.chat_input, place it AFTER the history container
        # The chat_input itself cannot be fixed at the very bottom of the viewport
        # but this places it below the scrollable history within the column layout.
        if math_prompt := st.chat_input("Ask a Mathematics question...", key="math_input"):
            # 1. Add user message to session state
            st.session_state.messages_math.append({"role": "user", "content": math_prompt})
            # No need to display here, the rerun will handle it inside the container

            # 2. Prepare for LangChain call (load history from LC memory)
            try:
                with st.spinner("Thinking..."):
                    # Load history from the dedicated LangChain memory
                    chat_history_math = st.session_state.mem_math.load_memory_variables({})['chat_history']
                    print(f"Chat history (Math): {chat_history_math}")  # Debugging line
                    if not isinstance(chat_history_math, list):
                        chat_history_math = [] # Fallback

                    # Create retriever and chains (as you have them in your code)
                    retriever_math = st.session_state.vectors_math.as_retriever()
                    document_chain_math = create_stuff_documents_chain(llm_math, prompt)
                    retrieval_chain_math = create_retrieval_chain(retriever_math, document_chain_math)

                    # Invoke chain
                    result_math = retrieval_chain_math.invoke({
                        "input": math_prompt,
                        "chat_history": chat_history_math
                    })
                    assistant_response = result_math["answer"]

                    # 3. Add assistant response to session state
                    st.session_state.messages_math.append({"role": "assistant", "content": assistant_response})
                    # No need to display here, the rerun will handle it inside the container

                    # 4. Save context to LangChain memory
                    st.session_state.mem_math.save_context(
                        {"input": math_prompt},
                        {"answer": assistant_response}
                    )
                    # Trigger a rerun to display the new messages in the container
                    st.rerun()

            except Exception as e:
                st.error(f"Error processing Math question: {e}")
                # Add error message to chat session state
                st.session_state.messages_math.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
                # Trigger a rerun to display the error message
                st.rerun()


with col2:
    st.markdown("### NCERT Science")
    # Ensure vector store is loaded before allowing questions
    if "vectors_science" not in st.session_state or st.session_state.vectors_science is None:
        st.warning("Please load Science data first using the sidebar button.")
    else:
        # Create a container for the chat history with a fixed height and scroll
        with st.container(height=500): # <--- Added container here
             # Display existing chat messages from session state INSIDE the container
            for message in st.session_state.messages_science:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"]) # Use markdown to render potential LaTeX

        # Accept user input using st.chat_input, place it AFTER the history container
        if science_prompt := st.chat_input("Ask a Science question...", key="science_input"):
            # 1. Add user message to session state
            st.session_state.messages_science.append({"role": "user", "content": science_prompt})
             # No need to display here, the rerun will handle it inside the container

            # 2. Prepare for LangChain call (load history from LC memory)
            try:
                with st.spinner("Thinking..."):
                    # Load history from the dedicated LangChain memory
                    chat_history_science = st.session_state.mem_science.load_memory_variables({})['chat_history']
                    print(f"Chat history (Science): {chat_history_science}")  # Debugging line
                    # Check if chat_history_science is a list, if not, fallback to empty list
                    if not isinstance(chat_history_science, list):
                        chat_history_science = []

                    # Create retriever and chains (as you have them in your code)
                    retriever_science = st.session_state.vectors_science.as_retriever()
                    document_chain_science = create_stuff_documents_chain(llm_science, prompt)
                    retrieval_chain_science = create_retrieval_chain(retriever_science, document_chain_science)

                    # Invoke chain
                    result_science = retrieval_chain_science.invoke({
                        "input": science_prompt,
                        "chat_history": chat_history_science
                    })
                    assistant_response = result_science["answer"]

                    # 3. Add assistant response to session state
                    st.session_state.messages_science.append({"role": "assistant", "content": assistant_response})
                    # No need to display here, the rerun will handle it inside the container

                    # 4. Save context to LangChain memory
                    st.session_state.mem_science.save_context(
                        {"input": science_prompt},
                        {"answer": assistant_response}
                    )
                    # Trigger a rerun to display the new messages in the container
                    st.rerun()

            except Exception as e:
                st.error(f"Error processing Science question: {e}")
                # Add error message to chat session state
                st.session_state.messages_science.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
                # Trigger a rerun to display the error message
                st.rerun()