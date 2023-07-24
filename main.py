import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import VertexAI

st.title("📈ChatFile")

uploaded_file=st.file_uploader("Upload your file!")
st.caption("Only CSV and PDF files are supported.")

if uploaded_file is not None:
    #LLM Operations
    llm = VertexAI()
    agent=create_csv_agent(llm, uploaded_file,verbose=True)
    
    # User Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt is not None
        response = f"Echo: {prompt}"
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})