import streamlit as st
import time
import KNearestNeighbor.KNN as KNN


st.title("K Nearest Neighbor 🌎🔄")

st.write(
    "This is a simple K Nearest Neighbor Model which allows you to search for documents in the Cranfield collection.")
query = st.text_input("Enter Query")
search_button = st.button("Search 🔍")
results = st.empty()
if search_button:
    result = KNN.predict(query)
    with st.spinner("Searching..."):
        time.sleep(1)
        if len(result) == 0:
            results.text("No results found 😞 ")
        else:
            results.text(result)
            st.balloons()
