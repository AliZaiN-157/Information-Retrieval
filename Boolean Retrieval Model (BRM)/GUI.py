import streamlit as st
import BRM
import time


def main():
    st.set_page_config(page_title="Boolean Retrieval Model",
                       page_icon="🔍", layout="centered")
    st.title("Boolean Retrieval Model 🔍 🧐")
    st.subheader("Ali Zain K21-4653 :computer:")
    query = st.text_input("Enter Query")
    search_button = st.button("Search 🔍")
    results = st.empty()
    if search_button:
        with st.spinner("Searching..."):
            time.sleep(1)
            st.balloons()
            result = BRM.queryFetcher(query)
            results.text(result)


if __name__ == '__main__':
    main()
