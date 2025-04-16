import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

#set the groq_api_key from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")

#set up the google generative AI API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# Design enhancements: Page config and custom CSS
st.set_page_config(page_title="NCERT Tutor", page_icon=":book:", layout="wide", initial_sidebar_state="expanded")

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
</style>
""", unsafe_allow_html=True)

st.title("NCERT Math and Science Tutor")
st.subheader("Ask Questions From NCERT Math and Science Textbooks")

llm_math=ChatGroq(groq_api_key=groq_api_key,
             model_name="gemma2-9b-it")
llm_science=ChatGroq(groq_api_key=groq_api_key,
             model_name="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
"""
Assume you are a Mathematics and Science teacher of very young childern and you need to answer the following question to a child. Explain the concept in a very simple way and provide examples and analogies to make it easy to understand. Answer only as a teacher who is explaining things to students. Take the following texts from NCERT Mathematics and Science textbooks and answer the question as a who teaches from NECERT textbooks. 
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding_math():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
        st.session_state.loader=PyPDFDirectoryLoader("./Mathematics") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        #st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200, separators= ["\n\n", "\n", " ", ""]) 
        st.session_state.semantic_chunker=SemanticChunker(st.session_state.embeddings, min_chunk_size=200) #chunking

        # Extract text contents from each Document object
        texts = [doc.page_content for doc in st.session_state.docs]


        ## Chunk Creation
        st.session_state.final_documents=st.session_state.semantic_chunker.create_documents(texts) #splitting

        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector Google embeddings
        print(st.session_state.final_documents[200])
        print("Total Number of Chunks Created: ", len(st.session_state.final_documents))
        print("Total Number of Documents: ", len(st.session_state.docs))


def vector_embedding_science():
    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
        st.session_state.loader=PyPDFDirectoryLoader("./Science") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        #st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200, separators= ["\n\n", "\n", " ", ""]) 
        st.session_state.semantic_chunker=SemanticChunker(st.session_state.embeddings , min_chunk_size=200)

        # Extract text contents from each Document object
        texts = [doc.page_content for doc in st.session_state.docs]


        ## Chunk Creation
        st.session_state.final_documents=st.session_state.semantic_chunker.create_documents(texts) #splitting

        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector Google embeddings
        print(st.session_state.final_documents[200])
        print("Total Number of Chunks Created: ", len(st.session_state.final_documents))
        print("Total Number of Documents: ", len(st.session_state.docs))


# Move data loading buttons into the sidebar with design enhancements
if st.sidebar.button("Load Math Data"):
    vector_embedding_math()
    st.sidebar.success("Vector Store DB Is Ready")

if st.sidebar.button("Load Science Data"):
    vector_embedding_science()
    st.sidebar.success("Science DB Is Ready")

# Organize question inputs into two columns for a better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### NCERT Mathematics")
    prompt1 = st.text_input("Enter Your Question From NCERT Mathematics book:", key='input_math')
    if st.button("Get Answer", key='get_answer_math'):
        # Create the retrieval chain
        document_chain = create_stuff_documents_chain(llm_math, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

with col2:
    st.markdown("### NCERT Science")
    prompt2 = st.text_input("Enter Your Question From NCERT Science book:", key='input_science')
    if st.button("Get Answer", key='get_answer_science'):
        # Create the retrieval chain
        document_chain = create_stuff_documents_chain(llm_science, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': prompt2})
        st.write(response['answer'])
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")




