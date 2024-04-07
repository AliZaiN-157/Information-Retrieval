import streamlit as st
import BooleanRetrievalModel.BRM as BRM
import VectorSpaceModel.VSM as VSM
import time


def main():
    st.set_page_config(page_title="Information Retrieval System",
                       page_icon="🔍", layout="centered")
    st.title("Information Retrieval System")
    st.subheader("Ali Zain K21-4653 💻")

    st.sidebar.empty()


if __name__ == '__main__':
    main()
