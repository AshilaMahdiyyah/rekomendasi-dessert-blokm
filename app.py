import streamlit as st
import pandas as pd
import numpy as np
import ast

st.title("Sistem Rekomendasi Dessert di Blok M")

# load data
@st.cache_data
def load_data():
    df = pd.read_csv("/content/drive/MyDrive/TA/dataset_with_category.csv")
    return df

df = load_data()

# ===== PREPROCESS LIST STRING =====
for col in ['menu_category', 'flavor_category']:
    df[col] = df[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

def clean_label(x):
    return x.replace("_", " ").title()

# ===== FILTER UI =====
st.sidebar.header("Filter Preferensi")

menu_flat = [item for sublist in df['menu_category'] for item in sublist]
menu_counts = pd.Series(menu_flat).value_counts()
menu_options = [clean_label(x) for x in menu_counts.index]

menu = st.sidebar.multiselect("Kategori Dessert", menu_options)

flavor_flat = [item for sublist in df['flavor_category'] for item in sublist]
flavor_counts = pd.Series(flavor_flat).value_counts()
flavor_options = [clean_label(x) for x in flavor_counts.index]

flavor = st.sidebar.multiselect("Flavor", flavor_options)

price = st.sidebar.selectbox(
    "Range Price",
    ["All"] + sorted(df['range_price'].dropna().unique().tolist())
)

dine = st.sidebar.selectbox(
    "Dine Option",
    ["All"] + sorted(df['dine_option'].dropna().unique().tolist())
)

rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 4.0)

# ===== BUTTON =====
if st.button("🔍 Cari Rekomendasi"):

    filtered_df = df.copy()

    if menu:
        filtered_df = filtered_df[
            filtered_df['menu_category'].apply(
                lambda x: any(m.lower().replace(" ", "_") in x for m in menu)
            )
        ]

    if flavor:
        filtered_df = filtered_df[
            filtered_df['flavor_category'].apply(
                lambda x: any(f.lower().replace(" ", "_") in x for f in flavor)
            )
        ]

    if price != "All":
        filtered_df = filtered_df[filtered_df['range_price'] == price]

    if dine != "All":
        filtered_df = filtered_df[filtered_df['dine_option'] == dine]

    filtered_df = filtered_df[filtered_df['avgRating'] >= rating]

    st.write("Jumlah kandidat:", len(filtered_df))

    top_result = filtered_df.sort_values(
        by="avgRating", ascending=False
    ).head(5)

    st.subheader("Top 5 Rekomendasi")
    st.dataframe(
        top_result[['nama_tempat','avgRating','range_price']]
    )
