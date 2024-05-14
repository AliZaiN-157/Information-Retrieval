import streamlit as st
import time
import KMeanClustering.KMean as KMean


st.title("K-Mean Clustering 🧩🌐")
slider = st.slider("Select Number of Clusters", 2, 10, 5)
results = st.empty()
if st.button("Cluster"):
    df, kmeans, wcss, sil, rand, purity, silhouette, RI = KMean.set_cluster(
        n_clusters=slider)
    with st.spinner("Clustering..."):
        time.sleep(1)
        st.metric(label="Purity", value=purity)
        st.metric(label="Silhouette", value=silhouette)
        st.metric(label="Rand Index", value=RI)
