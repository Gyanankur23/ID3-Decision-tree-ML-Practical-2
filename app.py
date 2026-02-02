import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="ID3 Tree Visualizer")

def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    return -sum((c/len(col)) * math.log2(c/len(col)) for c in counts)

def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals = df[attr].unique()
    weighted_entropy = sum((len(df[df[attr] == v]) / len(df)) * entropy(df[df[attr] == v][target]) for v in vals)
    return total_entropy - weighted_entropy

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    
    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}
    
    for val in df[best].unique():
        sub_df = df[df[best] == val]
        new_attrs = [a for a in attrs if a != best]
        tree[best][val] = id3(sub_df, target, new_attrs)
    return tree

st.title("ID3 Decision Tree Classifier")
st.info("This application implements the ID3 algorithm using Entropy and Information Gain. It constructs a decision tree to classify categorical data based on feature importance.")

st.sidebar.header("Configuration")
data_option = st.sidebar.selectbox("Dataset Source", ["Synthetic Tennis Data", "Upload CSV"])

if data_option == "Synthetic Tennis Data":
    data_dict = {
        "Outlook": ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'overcast', 'sunny', 'sunny', 'overcast', 'rain', 'overcast', 'overcast', 'rain', 'sunny'],
        "Humidity": ['high', 'normal', 'high', 'normal', 'high', 'high', 'normal', 'normal', 'normal', 'normal', 'normal', 'high', 'high', 'normal'],
        "Wind": ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong'],
        "PlayTennis": ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
    df = pd.DataFrame(data_dict)
else:
    file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

target_col = st.sidebar.selectbox("Select Target Label", df.columns, index=len(df.columns)-1)
features = [c for c in df.columns if c != target_col]

st.subheader("Data Preview")
st.dataframe(df.head(), use_container_width=True)

if st.button("Generate Decision Tree"):
    tree = id3(df, target_col, features)
    
    st.subheader("Tree Visualization (JSON Structure)")
    st.json(tree)
    
    st.subheader("Logic Flow")
    def display_tree(tree, indent=""):
        if not isinstance(tree, dict):
            st.text(f"{indent}Result: {tree}")
            return
        for node, branches in tree.items():
            st.text(f"{indent}[{node}]")
            for value, branch in branches.items():
                st.text(f"{indent}  --> {value}")
                display_tree(branch, indent + "      ")
    
    display_tree(tree)

st.sidebar.divider()
st.sidebar.write("Step 1: Select Data")
st.sidebar.write("Step 2: Define Target")
st.sidebar.write("Step 3: Build & Visualize")
