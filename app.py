
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

sns.set_style("whitegrid")


# ==========================================================
# TITLE
# ==========================================================

st.title("🛒 Market Basket Analysis")
st.subheader("Product Association and Recommendation System")

st.markdown("""
Discover frequently purchased products together using
**Apriori Algorithm** and **Association Rule Mining**.
""")

st.divider()


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    df = pd.read_excel("dataset/Online_Retail.xlsx")

    df.dropna(subset=["Description"], inplace=True)

    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]

    df = df[df["Quantity"] > 0]

    df = df[df["UnitPrice"] > 0]

    df["Description"] = (
        df["Description"]
        .str.strip()
        .str.upper()
    )

    df = df[df["Country"] == "United Kingdom"]

    return df


try:

    df = load_dataset()

except Exception as e:

    st.error(f"Error loading dataset: {e}")

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙️ Analysis Parameters")

min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.001,
    max_value=0.10,
    value=0.01,
    step=0.001
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.30,
    step=0.05
)

top_n_products = st.sidebar.slider(
    "Number of Products to Analyze",
    min_value=10,
    max_value=100,
    value=60,
    step=10
)

max_itemset_length = st.sidebar.slider(
    "Maximum Itemset Length",
    min_value=2,
    max_value=5,
    value=3
)

st.sidebar.markdown("---")

st.sidebar.info("""
**Support:** Frequency of an itemset in transactions.

**Confidence:** Probability of buying B when A is purchased.

**Lift:** Strength of association between products.
""")


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

st.header("📊 Dataset Overview")

total_transactions = df["InvoiceNo"].nunique()

total_products = df["Description"].nunique()

total_items = df["Quantity"].sum()

total_revenue = (
    df["Quantity"] * df["UnitPrice"]
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Unique Products",
    f"{total_products:,}"
)

col3.metric(
    "Items Sold",
    f"{total_items:,}"
)

col4.metric(
    "Total Revenue",
    f"£{total_revenue:,.0f}"
)


# ==========================================================
# EXPLORATORY DATA ANALYSIS
# ==========================================================

st.header("📈 Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs([
    "Top Products",
    "Transaction Analysis",
    "Dataset Preview"
])


# ----------------------------------------------------------
# TOP PRODUCTS
# ----------------------------------------------------------

with tab1:

    product_frequency = (
        df.groupby("Description")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        x=product_frequency.values,
        y=product_frequency.index,
        hue=product_frequency.index,
        palette="viridis",
        legend=False,
        ax=ax
    )

    ax.set_title("Top 15 Products by Transaction Frequency")

    ax.set_xlabel("Number of Transactions")

    ax.set_ylabel("Product")

    st.pyplot(fig)

    plt.close(fig)


# ----------------------------------------------------------
# TRANSACTION ANALYSIS
# ----------------------------------------------------------

with tab2:

    transaction_sizes = (
        df.groupby("InvoiceNo")["Description"]
        .count()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        transaction_sizes,
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title("Distribution of Items per Transaction")

    ax.set_xlabel("Number of Items")

    ax.set_ylabel("Number of Transactions")

    st.pyplot(fig)

    plt.close(fig)

    st.write("Average items per transaction:",
             round(transaction_sizes.mean(), 2))

    st.write("Maximum items in one transaction:",
             transaction_sizes.max())


# ----------------------------------------------------------
# DATASET PREVIEW
# ----------------------------------------------------------

with tab3:

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ==========================================================
# CREATE BASKET MATRIX
# ==========================================================

st.header("🧺 Transaction Basket")

@st.cache_data
def create_basket(data, top_n):

    basket = (
        data.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    basket = basket.astype(bool)

    product_frequency = basket.sum().sort_values(
        ascending=False
    )

    selected_products = product_frequency.head(
        top_n
    ).index

    basket = basket[selected_products]

    return basket


basket = create_basket(
    df,
    top_n_products
)

st.write(
    "Basket Matrix Shape:",
    basket.shape
)

st.caption(
    "Rows represent transactions and columns represent products. "
    "True indicates that a product was purchased."
)

with st.expander("View Transaction Basket"):

    st.dataframe(
        basket.head(20),
        use_container_width=True
    )


# ==========================================================
# APPLY APRIORI
# ==========================================================

st.header("🔍 Frequent Itemset Mining")

if st.button(
    "🚀 Run Apriori Algorithm",
    type="primary",
    use_container_width=True
):

    with st.spinner("Mining frequent itemsets..."):

        frequent_itemsets = apriori(
            basket,
            min_support=min_support,
            use_colnames=True,
            max_len=max_itemset_length
        )

        if frequent_itemsets.empty:

            st.warning(
                "No frequent itemsets found. "
                "Try decreasing minimum support."
            )

            st.stop()

        frequent_itemsets["itemsets"] = (
            frequent_itemsets["itemsets"]
            .apply(lambda x: ", ".join(sorted(list(x))))
        )

        frequent_itemsets = frequent_itemsets.sort_values(
            by="support",
            ascending=False
        )

        st.session_state["frequent_itemsets"] = (
            frequent_itemsets
        )

        # Generate Association Rules

        raw_itemsets = apriori(
            basket,
            min_support=min_support,
            use_colnames=True,
            max_len=max_itemset_length
        )

        rules = association_rules(
            raw_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        rules = rules[rules["lift"] > 1]

        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(list(x)))
        )

        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(list(x)))
        )

        rules = rules.sort_values(
            by="lift",
            ascending=False
        )

        st.session_state["rules"] = rules

        st.success("Apriori Analysis Completed Successfully!")


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

if "frequent_itemsets" in st.session_state:

    frequent_itemsets = (
        st.session_state["frequent_itemsets"]
    )

    rules = st.session_state["rules"]

    st.divider()

    st.header("📌 Mining Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Frequent Itemsets",
        len(frequent_itemsets)
    )

    col2.metric(
        "Association Rules",
        len(rules)
    )

    if not rules.empty:

        col3.metric(
            "Maximum Lift",
            round(rules["lift"].max(), 2)
        )

    # ------------------------------------------------------
    # FREQUENT ITEMSETS
    # ------------------------------------------------------

    st.subheader("Frequent Itemsets")

    st.dataframe(
        frequent_itemsets,
        use_container_width=True
    )

    # ------------------------------------------------------
    # ASSOCIATION RULES
    # ------------------------------------------------------

    st.subheader("Association Rules")

    if rules.empty:

        st.warning(
            "No rules generated. "
            "Try decreasing confidence."
        )

    else:

        st.dataframe(
            rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ],
            use_container_width=True
        )

    # ------------------------------------------------------
    # ASSOCIATION RULE VISUALIZATION
    # ------------------------------------------------------

    if not rules.empty:

        st.subheader("📊 Support vs Confidence")

        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            rules["support"],
            rules["confidence"],
            c=rules["lift"],
            cmap="viridis",
            s=80,
            alpha=0.7
        )

        plt.colorbar(
            scatter,
            label="Lift"
        )

        ax.set_xlabel("Support")

        ax.set_ylabel("Confidence")

        ax.set_title(
            "Association Rules Visualization"
        )

        st.pyplot(fig)

        plt.close(fig)

        # --------------------------------------------------
        # TOP RULES
        # --------------------------------------------------

        st.subheader("🏆 Top Association Rules by Lift")

        top_rules = rules.head(10).copy()

        top_rules["Rule"] = (
            top_rules["antecedents"]
            + " → "
            + top_rules["consequents"]
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        sns.barplot(
            data=top_rules,
            x="lift",
            y="Rule",
            hue="Rule",
            palette="magma",
            legend=False,
            ax=ax
        )

        ax.set_title("Top 10 Rules by Lift")

        st.pyplot(fig)

        plt.close(fig)

    # ======================================================
    # PRODUCT RECOMMENDATION
    # ======================================================

    st.divider()

    st.header("🛍️ Product Recommendation System")

    if not rules.empty:

        all_products = set()

        for product in rules["antecedents"]:

            all_products.update(
                product.split(", ")
            )

        for product in rules["consequents"]:

            all_products.update(
                product.split(", ")
            )

        selected_product = st.selectbox(
            "Select a product to get recommendations",
            sorted(all_products)
        )

        if st.button("Get Recommendations"):

            matching_rules = rules[
                rules["antecedents"].apply(
                    lambda x: selected_product in x.split(", ")
                )
            ]

            matching_rules = matching_rules.sort_values(
                by="lift",
                ascending=False
            )

            if matching_rules.empty:

                st.warning(
                    "No recommendations found for this product."
                )

            else:

                st.success(
                    f"Products frequently associated with "
                    f"{selected_product}"
                )

                st.dataframe(
                    matching_rules[
                        [
                            "antecedents",
                            "consequents",
                            "support",
                            "confidence",
                            "lift"
                        ]
                    ],
                    use_container_width=True
                )

    # ======================================================
    # DOWNLOAD RESULTS
    # ======================================================

    st.divider()

    st.header("📥 Download Results")

    col1, col2 = st.columns(2)

    col1.download_button(
        "Download Frequent Itemsets CSV",
        frequent_itemsets.to_csv(index=False),
        file_name="frequent_itemsets.csv",
        mime="text/csv"
    )

    col2.download_button(
        "Download Association Rules CSV",
        rules.to_csv(index=False),
        file_name="association_rules.csv",
        mime="text/csv"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
<div style="text-align:center">

**Market Basket Analysis using Apriori Algorithm**

Data Mining Lab Project

</div>
""", unsafe_allow_html=True)
