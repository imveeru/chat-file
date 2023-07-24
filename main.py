import streamlit as st

st.title("📈ChatFile")

uploaded_file=st.file_uploader("Upload your file!")
st.caption("Only CSV and PDF files are supported.")

if uploaded_file is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = f"Echo: {prompt}"
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})