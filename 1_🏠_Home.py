import streamlit as st


def main():
    st.set_page_config(page_title="Information Retrieval System",
                       page_icon="🔍", layout="wide")

    custom_html = """
    <div class="banner">
        <img src="https://miro.medium.com/v2/resize:fit:852/1*JTvMfiXgaO4LGLI3Lj2dsw.jpeg" 
        alt="Banner Image">
    </div>
    <style>
        .body {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        .banner {
            width: 100%;
            height: 200px;
        }
        .banner img {
            width: 100%;
            height: 500px
        }
    </style>
    """
    # Display the custom HTML
    with st.container(height=500, border=False):
        st.markdown(f"{custom_html}", unsafe_allow_html=True)
        # st.components.v1.html(custom_html)

    st.markdown("<h2><center> Information Retrieval System </center> </h2>",
                unsafe_allow_html=True)
    st.markdown("<h3><center> Ali Zain K21-4653 💻 </center> </h3>",
                unsafe_allow_html=True)

    container = st.container(border=False)
    with container:
        st.write(
            "This project is a simple implementation of Information Retrieval System using different models.")
        st.write(
            "The dataset used in this project is a collection of research papers from the field of Computer Science.")
        st.write("The project is divided into two parts:")
        container.page_link("pages/2_🔍_Boolean Retrieval Model.py",
                            label="Boolean Retrieval Model", icon="🔍", use_container_width=True)
        container.page_link("pages/3_🚀_Vector Space Model.py",
                            label="Vector Space Model", icon="🚀",  use_container_width=True)
        container.page_link("pages/4_🔬_K Nearest Neighbor.py",
                            label="K Nearest Neighbor Model", icon="🔬",  use_container_width=True)
        container.page_link("pages/5_🧩_K Mean Clustering.py",
                            label="K Mean Clustering", icon="🧩",  use_container_width=True)


if __name__ == '__main__':

    main()
