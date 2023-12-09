import streamlit as st
import pandas as pd
import plotly.express as px
import io
from sklearn.cluster import KMeans
import plotly.graph_objects as go

st.set_page_config(page_title="DWM MP", layout="wide") 
st.markdown(""" <style> .reportview-container { margin-top: -2em; } #MainMenu {visibility: hidden;} .stDeployButton {display:none;} footer {visibility: hidden;} #stDecoration {display:none;} </style> """, unsafe_allow_html=True)

def df_info(df):
    df.columns = df.columns.str.replace(' ', '_')
    buffer = io.StringIO() 
    df.info(buf=buffer)
    s = buffer.getvalue() 
    df_info = s.split('\n')
    counts = []
    names = []
    nn_count = []
    dtype = []
    for i in range(5, len(df_info)-3):
        line = df_info[i].split()
        counts.append(line[0])
        names.append(line[1])
        nn_count.append(line[2])
        dtype.append(line[4])
    df_info_dataframe = pd.DataFrame(data = {'#':counts, 'Column':names, 'Non-Null Count':nn_count, 'Data Type':dtype})
    return df_info_dataframe.drop('#', axis = 1)

def df_isnull(df):
    res = pd.DataFrame(df.isnull().sum()).reset_index()
    res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
    res['Percentage'] = res['Percentage'].astype(str) + '%'
    return res.rename(columns={'index': 'Column', 0: 'Number of null values'})

def number_of_outliers(df):
    df = df.select_dtypes(exclude = 'object')
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    ans = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    df = pd.DataFrame(ans).reset_index().rename(columns = {'index':'column', 0:'count_of_outliers'})
    return df

def space(num_lines=1):
    for _ in range(num_lines):
        st.write("")

def sidebar_space(num_lines=1):
    for _ in range(num_lines):
        st.sidebar.write("")

def sidebar_multiselect_container(massage, arr, key):
    container = st.sidebar.container()
    select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default = list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default = arr[0])
    return selected_num_cols 

def perform_kmeans(df, num_clusters, features):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[features])
    return df

def visualize_kmeans(df, features, cluster_column):
    if len(features) >= 2:
        cluster_colors = px.colors.qualitative.Set1[:len(df[cluster_column].unique())]
        fig = go.Figure()
        for cluster, color in zip(df[cluster_column].unique(), cluster_colors):
            cluster_data = df[df[cluster_column] == cluster]
            fig.add_trace(go.Scatter(
                x=cluster_data[features[0]],
                y=cluster_data[features[1]],
                mode='markers',
                marker=dict(color=color),
                name=f'Cluster {cluster}'
            ))
        fig.update_layout(title='K-means Clustering Visualization', xaxis_title=features[0], yaxis_title=features[1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select two features for K-means clustering visualization.")
st.header("DWM Mini Project")
st.write('<p style="font-size:100%">&nbsp Team Members:</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 1. Amey Rane (60003210163)</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 2. Varad Patil (60003210166)</p>', unsafe_allow_html=True)
st.write('<p style="font-size:100%">&nbsp 3. Rahul Chougule (60003210167)</p>', unsafe_allow_html=True)
space()
st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)
dataset = st.file_uploader(label = '')
if dataset:
    df = pd.read_csv(dataset)
    st.subheader('Dataframe:')
    n, m = df.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
    st.dataframe(df)
    all_vizuals = ['Info', 'Null Value Info', 'Data Exploration', 'Target Analysis', 
                   'Distribution of Numerical Columns', 'Count Plots of Categorical Columns', 
                   'Box Plots', 'Outlier Analysis', 'Variance of Target with Categorical Columns', 'K-means Clustering']
    sidebar_space(3)         
    vizuals = st.sidebar.multiselect("Choose visualizations:", all_vizuals)
    if 'Info' in vizuals:
        st.subheader('Info:')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(df_info(df))
    if 'Null Value Info' in vizuals:
        st.subheader('Null Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There are no null values in your dataset.')
        else:
            null_handling_option = st.radio("Choose null values handling option:", ("Drop", "Replace with mean", "Replace with 0"))
            c1, c2, c3 = st.columns([0.5, 2, 0.5])
            na_info_df = df_isnull(df)
            c2.dataframe(na_info_df, width=1500)
            if null_handling_option == "Replace with Mean":
                numeric_columns = df.select_dtypes(include=['number']).columns
                df.fillna(df[numeric_columns].mean(), inplace=True)
                st.success("Null values have been filled with mean in the dataset.")
            elif null_handling_option == "Drop":
                df.dropna(inplace=True)
                st.success("Null values have been dropped from the dataset.")
            else:
                numeric_columns = df.select_dtypes(include=['number']).columns
                df.fillna(0, inplace=True)
                st.success("Null values have been filled with mean in the dataset.")
    if 'Data Exploration' in vizuals:
        st.subheader('Data Exploration:')
        st.dataframe(df.describe())
    if 'Target Analysis' in vizuals:
        st.subheader("Select column for Histogram:")    
        target_column = st.selectbox("", df.columns, index = len(df.columns) - 1)
        st.subheader("Histogram")
        fig = px.histogram(df, x = target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)
    num_columns = df.select_dtypes(exclude = 'object').columns
    cat_columns = df.select_dtypes(include = 'object').columns
    if 'Distribution of Numerical Columns' in vizuals:
        if len(num_columns) == 0:
            st.write('There are no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Distribution plots:', num_columns, 'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:
                    if (i >= len(selected_num_cols)):
                        break
                    fig = px.histogram(df, x = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1
    if 'Count Plots of Categorical Columns' in vizuals:
        if len(cat_columns) == 0:
            st.write('There are no categorical columns in the data.')
        else:
            selected_cat_cols = sidebar_multiselect_container('Choose columns for Count plots:', cat_columns, 'Count')
            st.subheader('Count plots of categorical columns')
            i = 0
            while (i < len(selected_cat_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:
                    if (i >= len(selected_cat_cols)):
                        break
                    fig = px.histogram(df, x = selected_cat_cols[i], color_discrete_sequence=['indianred'])
                    j.plotly_chart(fig)
                    i += 1
    if 'Box Plots' in vizuals:
        if len(num_columns) == 0:
            st.write('There are no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Box plots:', num_columns, 'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:
                    if (i >= len(selected_num_cols)):
                        break
                    fig = px.box(df, y = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1
    if 'Outlier Analysis' in vizuals:
        st.subheader('Outlier Analysis')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(number_of_outliers(df))
    if 'Variance of Target with Categorical Columns' in vizuals:
        df_1 = df.dropna()
        high_cardi_columns = []
        normal_cardi_columns = []
        for i in cat_columns:
            if (df[i].nunique() > df.shape[0] / 10):
                high_cardi_columns.append(i)
            else:
                normal_cardi_columns.append(i)
        if len(normal_cardi_columns) == 0:
            st.write('There are no categorical columns with normal cardinality in the data.')
        else:
            st.subheader('Variance of target variable with categorical columns')
            selected_cat_cols = sidebar_multiselect_container('Choose columns for Category Colored plots:', normal_cardi_columns, 'Category')
            if 'Target Analysis' not in vizuals:   
                target_column = st.selectbox("Select target column:", df.columns, index = len(df.columns) - 1)
            i = 0
            while (i < len(selected_cat_cols)):
                fig = px.histogram(df_1, color = selected_cat_cols[i], x = target_column)

                st.plotly_chart(fig, use_container_width = True)
                i += 1
    if 'K-means Clustering' in vizuals:
        st.subheader('K-means Clustering')
        num_clusters = st.slider('Select the number of clusters:', min_value=2, max_value=10, value=2)
        features_for_clustering = sidebar_multiselect_container('Choose features for clustering:', num_columns, 'Clustering')
        df_clustered = perform_kmeans(df.copy(), num_clusters, features_for_clustering)
        visualize_kmeans(df_clustered, features_for_clustering, 'Cluster')