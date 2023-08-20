import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import VertexAI
import pandas as pd
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

import google.generativeai as palm
from google.auth import credentials
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel

st.set_page_config(
    page_title="ChatFile",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

#hide streamlit default
hide_st_style ='''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("📈ChatFile")

config = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
service_account_info=json.loads(config)
service_account_info["private_key"]=service_account_info["private_key"].replace("\\n","\n")

my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)
# Initialize Google AI Platform with project details and credentials
aiplatform.init(
    credentials=my_credentials,
)
project_id = service_account_info["project_id"]

vertexai.init(project=project_id, location="us-central1")

uploaded_file=st.file_uploader("Upload your file!")
st.caption("Only CSV and PDF files are supported.")

if uploaded_file is not None:
    #handling file format
    extension=os.path.splitext(uploaded_file.name)[1]
    
    if extension==".csv":
        df=pd.read_csv(uploaded_file)
        #LLM Operations
        with st.spinner("Reading the uploaded file..."):
            llm = VertexAI()
            agent=create_pandas_dataframe_agent(llm, df,verbose=True)
    elif extension==".pdf":
        pdf_reader=PdfReader(uploaded_file)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        #split into chunks
        text_splitter=CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=250,
            length_function=len
        )
        
        chunks=text_splitter.split_text(text)
        
        embeddings = VertexAIEmbeddings()
        
        knowledge_base=FAISS.from_texts(chunks,embeddings)

        
    else:
        st.error("Only CSV or PDF files can be uploaded")
        st.stop()
    
    #welcome message   
    with st.chat_message("assistant"):
            st.markdown(f"👋Hello! Ask me your questions related to the uploaded {extension.upper()[1:]} file.")
    # User Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"],unsafe_allow_html=True)

    if prompt := st.chat_input("Ask your question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt is not None and prompt != "":
            with st.spinner("Generating response..."):
                if extension==".csv":
                    response = agent.run(prompt)
                elif extension==".pdf":
                    docs=knowledge_base.similarity_search(prompt)
                    #st.write(docs)
                    llm=VertexAI()
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response=chain.run(input_documents=docs, question=prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})