
import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity


# LOAD DATA
df_menu = pd.read_csv("dataset_with_menu_recommendation (1).csv")


# LOAD MODEL
tfidf = pickle.load(open("tfidf.pkl", "rb"))
item_profile = pickle.load(open("item_profile.pkl", "rb"))


def get_rekomendasi(menu, flavor, price, dine, rating, top_n=10):

    user_query = f"{menu} {flavor}"
    user_vec = tfidf.transform([user_query])

    df = df_menu.copy()

    df["similarity"] = cosine_similarity(
        item_profile,
        user_vec
    ).flatten()

    df["rating_norm"] = df["avgRating"] / 5.0

    df["price_match"] = (df["range_price"] == price).astype(int)

    if dine == "both":
        df["dine_match"] = 1
    else:
        df["dine_match"] = df["dine_option"].apply(
            lambda x: 1 if x == dine or x == "both" else 0
        )

    df["combined_features"] = (
        df["similarity"] +
        df["rating_norm"] +
        df["price_match"] +
        df["dine_match"]
    ) / 4

    df = df[df["avgRating"] >= rating]

    if price is not None:
        df = df[df["range_price"] == price]

    if dine != "both":
        df = df[
            (df["dine_option"] == dine) |
            (df["dine_option"] == "both")
        ]

    df = df.sort_values(
        by=["combined_features", "similarity", "avgRating"],
        ascending=[False, False, False]
    )

    df = df.drop_duplicates(subset="nama_tempat")

    df = df.head(top_n)

    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1

    return df[[
        "rank",
        "nama_tempat",
        "recommended_menu",
        "avgRating",
        "range_price",
        "dine_option",
        "similarity",
        "combined_features"
    ]]
