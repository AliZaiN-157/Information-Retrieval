import streamlit as st
import VectorSpaceModel.VSM as VSM
import time


st.title("Vector Space Model 🔍🧐")
st.write(
    "This is a simple Vector Space Model which allows you to search for documents in the Cranfield collection.")
query = st.text_input("Enter Query", key="vsm_query")
search_button = st.button("Search 🔍", key="vsm_btn")
results = st.empty()
if search_button:
    with st.spinner("Searching..."):
        time.sleep(1)
        st.balloons()
        result = VSM.queryFetcher(query)
        results.text(result)
