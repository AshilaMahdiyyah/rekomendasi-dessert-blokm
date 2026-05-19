import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# LOAD DATA & MODEL
# =====================================================

df_menu = pd.read_csv("dataset_final_1.csv")

tfidf = joblib.load("tfidf.pkl")

item_profile = joblib.load("item_profile.pkl")

# =====================================================
# FUNCTION REKOMENDASI
# =====================================================

def get_rekomendasi(
    menu,
    flavor,
    price,
    dine,
    rating,
    top_n=10
):

    # =====================================================
    # 1. USER QUERY
    # =====================================================

    user_query = f"{menu} {flavor}"

    user_vec = tfidf.transform([user_query])

    df = df_menu.copy()

    # =====================================================
    # 2. HITUNG SIMILARITY
    # =====================================================

    df["similarity"] = cosine_similarity(
        item_profile,
        user_vec
    ).flatten()

    # =====================================================
    # 3. NORMALISASI FEATURE
    # =====================================================

    df["rating_norm"] = df["avgRating"] / 5.0

    # price match
    df["price_match"] = (
        df["range_price"] == price
    ).astype(int)

    # dine match
    if dine == "both":
        df["dine_match"] = 1
    else:
        df["dine_match"] = df["dine_option"].apply(
            lambda x: 1 if x == dine or x == "both" else 0
        )

    # =====================================================
    # 4. COMBINED SCORE
    # =====================================================

    df["combined_features"] = (
        df["similarity"] +
        df["rating_norm"] +
        df["price_match"] +
        df["dine_match"]
    ) / 4

    # =====================================================
    # 5. FILTERING
    # =====================================================

    # filter rating
    df = df[df["avgRating"] >= rating]

    # filter price
    if price is not None:
        df = df[df["range_price"] == price]

    # filter dine
    if dine != "both":
        df = df[
            (df["dine_option"] == dine) |
            (df["dine_option"] == "both")
        ]

    # =====================================================
    # 6. SORTING
    # =====================================================

    df = df.sort_values(
        by=[
            "combined_features",
            "similarity",
            "avgRating"
        ],
        ascending=[False, False, False]
    )

    # hapus duplikat tempat
    df = df.drop_duplicates(
        subset="nama_tempat"
    )

    # ambil top N
    df = df.head(top_n)

    # =====================================================
    # 7. OUTPUT
    # =====================================================

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
