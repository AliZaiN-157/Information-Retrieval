import streamlit as st
import BooleanRetrievalModel.BRM as BRM
import time


st.title("Boolean Retrieval Model 🔍🧐")
st.write("This is a simple Boolean Retrieval Model which allows you to search for documents in the Cranfield collection.")
query = st.text_input("Enter Query", key="brm_query")
search_button = st.button("Search 🔍", key="brm_btn")
results = st.empty()
if search_button:
    with st.spinner("Searching..."):
        time.sleep(1)
        st.balloons()
        result = BRM.queryFetcher(query)
        results.text(result)
