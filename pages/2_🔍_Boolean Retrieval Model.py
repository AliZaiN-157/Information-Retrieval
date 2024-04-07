import streamlit as st
import BooleanRetrievalModel.BRM as BRM
import time


st.title("Boolean Retrieval Model 🔍🧐")
st.write("This is a simple Boolean Retrieval Model which allows you to search for documents in the Cranfield collection.")
query = st.text_input("Enter Query")
search_button = st.button("Search 🔍")
results = st.empty()
if search_button:
    result = BRM.queryFetcher(query)
    with st.spinner("Searching..."):
        time.sleep(1)
        if len(result) == 0:
            results.text("No results found 😞")
        else:
            results.text(result)
            st.balloons()
