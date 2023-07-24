import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import VertexAI
import pandas as pd

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

uploaded_file=st.file_uploader("Upload your file!")
st.caption("Only CSV and PDF files are supported.")

if uploaded_file is not None:
    #LLM Operations
    df=pd.read_csv(uploaded_file)
    #st.dataframe(df)
    with st.spinner("Reading the uploaded file..."):
        llm = VertexAI()
        agent=create_pandas_dataframe_agent(llm, df,verbose=True)
    
    # User Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt is not None and prompt != "":
            with st.spinner("Generating response..."):
                response = agent.run(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})