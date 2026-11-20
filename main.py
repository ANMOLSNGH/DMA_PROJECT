


import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")


DATASET_PATH = "dataset/Online_Retail.xlsx"
OUTPUT_DIR = "outputs"

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.30

# To reduce computational cost, Apriori will be applied
# to the top frequent products.
TOP_N_PRODUCTS = 60

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_data():

    print("\nLoading dataset...")

    df = pd.read_excel(DATASET_PATH)

    print("\nOriginal Dataset Shape:", df.shape)

    print("\nFirst Five Rows:")
    print(df.head())

    return df


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

def preprocess_data(df):

    print("\nPerforming Data Preprocessing...")

    # Remove rows having missing product descriptions
    df = df.dropna(subset=["Description"])

    # Remove cancelled transactions
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove transactions having non-positive quantities
    df = df[df["Quantity"] > 0]

    # Remove invalid prices
    df = df[df["UnitPrice"] > 0]

    # Convert description to uppercase
    df["Description"] = df["Description"].str.strip().str.upper()

    # Keep only UK transactions
    df = df[df["Country"] == "United Kingdom"]

    print("\nCleaned Dataset Shape:", df.shape)

    print("\nNumber of Unique Transactions:",
          df["InvoiceNo"].nunique())

    print("\nNumber of Unique Products:",
          df["Description"].nunique())

    return df


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

def exploratory_analysis(df):

    print("\nPerforming Exploratory Data Analysis...")

    # --------------------------------------------------------
    # TOP 10 PRODUCTS
    # --------------------------------------------------------

    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        x=top_products.values,
        y=top_products.index,
        hue=top_products.index,
        palette="viridis",
        legend=False
    )

    plt.title("Top 10 Most Purchased Products")
    plt.xlabel("Total Quantity Sold")
    plt.ylabel("Product")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/top_products.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------------
    # TRANSACTION FREQUENCY
    # --------------------------------------------------------

    transaction_counts = df.groupby("InvoiceNo").size()

    plt.figure(figsize=(10, 6))

    sns.histplot(
        transaction_counts,
        bins=30,
        kde=True
    )

    plt.title("Distribution of Products per Transaction")
    plt.xlabel("Number of Products")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/transaction_distribution.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------------
    # TOP PRODUCTS BY TRANSACTION FREQUENCY
    # --------------------------------------------------------

    product_frequency = (
        df.groupby("Description")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        x=product_frequency.values,
        y=product_frequency.index,
        hue=product_frequency.index,
        palette="magma",
        legend=False
    )

    plt.title("Top 15 Products by Transaction Frequency")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Product")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/product_frequency.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------------

    print("\n========== DATASET SUMMARY ==========")

    print("Total Transactions:",
          df["InvoiceNo"].nunique())

    print("Total Products:",
          df["Description"].nunique())

    print("Total Items Sold:",
          df["Quantity"].sum())

    print("Average Items per Transaction:",
          round(transaction_counts.mean(), 2))

    print("Maximum Items in a Transaction:",
          transaction_counts.max())

    print("====================================")


# ============================================================
# 4. CONVERT DATA INTO TRANSACTION BASKET
# ============================================================

def create_basket(df):

    print("\nCreating Transaction Basket...")

    # Count product occurrence in each transaction
    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    # Convert quantity into binary values
    # 1 = Product purchased
    # 0 = Product not purchased

    basket = basket.map(lambda x: 1 if x > 0 else 0)

    print("\nBasket Matrix Shape:", basket.shape)

    # --------------------------------------------------------
    # FILTER TOP FREQUENT PRODUCTS
    # --------------------------------------------------------

    product_frequency = basket.sum().sort_values(ascending=False)

    top_products = product_frequency.head(
        TOP_N_PRODUCTS
    ).index

    basket = basket[top_products]

    print("\nBasket Matrix After Product Filtering:",
          basket.shape)

    return basket


# ============================================================
# 5. APPLY APRIORI ALGORITHM
# ============================================================

def generate_frequent_itemsets(basket):

    print("\nApplying Apriori Algorithm...")

    frequent_itemsets = apriori(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=3
    )

    # Sort by support
    frequent_itemsets = frequent_itemsets.sort_values(
        by="support",
        ascending=False
    )

    print("\n========== FREQUENT ITEMSETS ==========")

    print(frequent_itemsets.head(20))

    print("\nTotal Frequent Itemsets:",
          len(frequent_itemsets))

    # Convert frozensets into readable strings
    frequent_itemsets["itemsets"] = (
        frequent_itemsets["itemsets"]
        .apply(lambda x: ", ".join(list(x)))
    )

    frequent_itemsets.to_csv(
        f"{OUTPUT_DIR}/frequent_itemsets.csv",
        index=False
    )

    return frequent_itemsets


# ============================================================
# 6. GENERATE ASSOCIATION RULES
# ============================================================

def generate_association_rules(basket):

    print("\nGenerating Association Rules...")

    # First calculate frequent itemsets
    frequent_itemsets = apriori(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=3
    )

    if frequent_itemsets.empty:

        print("No frequent itemsets found.")

        return pd.DataFrame()

    # Generate rules using confidence
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    if rules.empty:

        print("No association rules found.")

        return rules

    # --------------------------------------------------------
    # FILTER RULES
    # --------------------------------------------------------

    rules = rules[
        (rules["lift"] > 1)
    ]

    # Sort by Lift
    rules = rules.sort_values(
        by="lift",
        ascending=False
    )

    # --------------------------------------------------------
    # CONVERT ITEMSETS TO READABLE STRINGS
    # --------------------------------------------------------

    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )

    rules["consequents"] = rules["consequents"].apply(
        lambda x: ", ".join(list(x))
    )

    print("\n========== ASSOCIATION RULES ==========")

    print(
        rules[
            [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]
        ].head(20)
    )

    print("\nTotal Association Rules:",
          len(rules))

    rules.to_csv(
        f"{OUTPUT_DIR}/association_rules.csv",
        index=False
    )

    return rules


# ============================================================
# 7. VISUALIZE FREQUENT ITEMSETS
# ============================================================

def visualize_frequent_itemsets(frequent_itemsets):

    print("\nVisualizing Frequent Itemsets...")

    if frequent_itemsets.empty:

        print("No itemsets available.")

        return

    # Sort by support
    top_itemsets = frequent_itemsets.sort_values(
        by="support",
        ascending=False
    ).head(15)

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_itemsets,
        x="support",
        y="itemsets",
        hue="itemsets",
        palette="viridis",
        legend=False
    )

    plt.title("Top Frequent Itemsets")
    plt.xlabel("Support")
    plt.ylabel("Itemsets")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/frequent_itemsets.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 8. VISUALIZE ASSOCIATION RULES
# ============================================================

def visualize_association_rules(rules):

    print("\nVisualizing Association Rules...")

    if rules.empty:

        print("No rules available.")

        return

    # --------------------------------------------------------
    # SUPPORT VS CONFIDENCE
    # --------------------------------------------------------

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="viridis",
        s=100,
        alpha=0.7
    )

    plt.colorbar(scatter, label="Lift")

    plt.xlabel("Support")
    plt.ylabel("Confidence")

    plt.title(
        "Association Rules: Support vs Confidence"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/association_rules_scatter.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------------
    # TOP 10 RULES BY LIFT
    # --------------------------------------------------------

    top_rules = rules.sort_values(
        by="lift",
        ascending=False
    ).head(10)

    top_rules["rule"] = (
        top_rules["antecedents"]
        + " → "
        + top_rules["consequents"]
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_rules,
        x="lift",
        y="rule",
        hue="rule",
        palette="magma",
        legend=False
    )

    plt.title("Top 10 Association Rules by Lift")
    plt.xlabel("Lift")
    plt.ylabel("Association Rule")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/top_rules_lift.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 9. PRODUCT ASSOCIATION NETWORK
# ============================================================

def create_network_graph(rules):

    print("\nCreating Product Association Network...")

    if rules.empty:

        print("No rules available.")

        return

    G = nx.DiGraph()

    # Select top rules
    top_rules = rules.sort_values(
        by="lift",
        ascending=False
    ).head(30)

    for _, row in top_rules.iterrows():

        antecedent = row["antecedents"]
        consequent = row["consequents"]

        G.add_edge(
            antecedent,
            consequent,
            weight=row["lift"]
        )

    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.5
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2500,
        node_color="skyblue"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=20,
        width=2
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_weight="bold"
    )

    plt.title(
        "Product Association Network"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/association_network.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 10. PRODUCT RECOMMENDATION SYSTEM
# ============================================================

def recommend_products(rules):

    print("\n========== PRODUCT RECOMMENDATION ==========")

    if rules.empty:

        print("No rules available.")

        return

    # Keep a copy of readable rules
    rules_copy = rules.copy()

    # Display available products
    all_products = set()

    for item in rules_copy["antecedents"]:

        all_products.update(
            item.split(", ")
        )

    for item in rules_copy["consequents"]:

        all_products.update(
            item.split(", ")
        )

    print("\nSome Available Products:")

    for product in sorted(all_products)[:30]:

        print("-", product)

    user_product = input(
        "\nEnter a product name: "
    ).strip().upper()

    # --------------------------------------------------------
    # FIND RULES WHERE PRODUCT IS IN ANTECEDENT
    # --------------------------------------------------------

    matching_rules = rules_copy[
        rules_copy["antecedents"].str.upper()
        .str.contains(user_product, regex=False)
    ]

    if matching_rules.empty:

        print("\nNo recommendations found for this product.")

        return

    matching_rules = matching_rules.sort_values(
        by="lift",
        ascending=False
    )

    print("\nRecommended Products:")

    print(
        matching_rules[
            [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]
        ].head(10).to_string(index=False)
    )


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)

    print(
        "MARKET BASKET ANALYSIS USING APRIORI ALGORITHM"
    )

    print("=" * 60)

    # Step 1: Load Dataset
    df = load_data()

    # Step 2: Preprocess Dataset
    df = preprocess_data(df)

    # Step 3: Exploratory Data Analysis
    exploratory_analysis(df)

    # Step 4: Create Basket Matrix
    basket = create_basket(df)

    # Step 5: Apply Apriori
    frequent_itemsets = generate_frequent_itemsets(
        basket
    )

    # Step 6: Generate Association Rules
    rules = generate_association_rules(
        basket
    )

    # Step 7: Visualize Frequent Itemsets
    visualize_frequent_itemsets(
        frequent_itemsets
    )

    # Step 8: Visualize Rules
    visualize_association_rules(
        rules
    )

    # Step 9: Network Graph
    create_network_graph(
        rules
    )

    # Step 10: Product Recommendations
    recommend_products(
        rules
    )

    print("\n" + "=" * 60)

    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")

    print("All output files are saved inside the outputs folder.")

    print("=" * 60)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
