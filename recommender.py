
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# LOAD DATA
df_menu = pd.read_csv("dataset_with_menu_recommendation.csv")

# LOAD MODEL
tfidf = pickle.load(open("tfidf.pkl", "rb"))
item_profile = pickle.load(open("item_profile.pkl", "rb"))
mlb_menu = pickle.load(open("mlb_menu.pkl", "rb"))
mlb_flavor = pickle.load(open("mlb_flavor.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


def get_rekomendasi(menu, flavor, price, dine, rating, top_n=10):

    # user query
    user_query = f"{menu} {flavor}"
    user_vec = tfidf.transform([user_query])

    df = df_menu.copy().reset_index(drop=True)

    # similarity
    sim = cosine_similarity(item_profile, user_vec).flatten()

    # feature engineering
    df["rating_norm"] = df["avgRating"] / 5
    df["price_match"] = (df["range_price"] == price).astype(int)

    if dine == "both":
        df["dine_match"] = 1
    else:
        df["dine_match"] = df["dine_option"].apply(
            lambda x: 1 if x == dine or x == "both" else 0
        )

    df["combined"] = (
        df["similarity"] +
        df["rating_norm"] +
        df["price_match"] +
        df["dine_match"]
    ) / 4

    # filtering
    df = df[df["avgRating"] >= rating]

    if price:
        df = df[df["range_price"] == price]

    if dine != "both":
        df = df[(df["dine_option"] == dine) | (df["dine_option"] == "both")]

    # ranking
    df = df.sort_values(
        by=["combined", "similarity", "avgRating"],
        ascending=False
    )

    df = df.drop_duplicates("nama_tempat")

    df = df.head(top_n).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df
