import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def prepare_kmeans_data(df):
    df_grouped = df.groupby("CustomerID").agg({
        "InvoiceNo": "nunique",
        "Total_Prix": "sum",
        "InvoiceDate": "max"
    }).reset_index()

    df_grouped.columns = ["CustomerID", "Frequency", "Monetary", "LastPurchase"]

    df_grouped["Recency"] = (
        pd.to_datetime(df["InvoiceDate"]).max()
        - pd.to_datetime(df_grouped["LastPurchase"])
    ).dt.days

    X = df_grouped[["Frequency", "Monetary", "Recency"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_grouped, X_scaled


def load_uploaded_file(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="latin1")


def recommend_products(rules, user_basket, top_n=5,
                       min_confidence=0.7, min_lift=1.5):
    """
    Génère des recommandations à partir d'un DataFrame de règles d'association.

    rules : DataFrame contenant au minimum les colonnes
            ['antecedents', 'consequents', 'confidence', 'lift']
    user_basket : set ou iterable de produits (chaînes). Comparaison insensible à la casse.
    top_n : nombre maximal de recommandations renvoyées
    min_confidence, min_lift : seuils pour filtrer les règles

    Retourne une liste de dictionnaires :
      [{'product': str, 'score': float, 'confidence': float, 'lift': float}, ...]
    """
    if rules is None or rules.empty:
        return []

    # Filtrer par confiance et lift
    rules_f = rules.copy()
    if 'confidence' in rules_f.columns:
        rules_f = rules_f[rules_f['confidence'] >= min_confidence]
    if 'lift' in rules_f.columns:
        rules_f = rules_f[rules_f['lift'] >= min_lift]

    if rules_f.empty:
        return []

    # Score simple pour trier : confidence * lift
    rules_f['score'] = rules_f.get('confidence', 0) * rules_f.get('lift', 0)
    rules_f = rules_f.sort_values(by='score', ascending=False)

    # Normaliser le panier utilisateur
    user_basket_set = set([str(x).strip().lower() for x in user_basket])

    recs = []
    for _, row in rules_f.iterrows():
        ants = row.get('antecedents')
        cons = row.get('consequents')

        # Convertir antecedents/consequents en set de strings (lowercase)
        def to_set(x):
            if x is None:
                return set()
            if isinstance(x, (set, frozenset, list, tuple)):
                return set(str(i).strip().lower() for i in x)
            # fallback: string
            return set(str(x).strip().lower().split(','))

        ants_set = to_set(ants)
        cons_set = to_set(cons)

        if ants_set and ants_set.issubset(user_basket_set):
            for p in cons_set:
                if p and p not in user_basket_set:
                    recs.append((p, float(row.get('score', 0)), float(row.get('confidence', 0)), float(row.get('lift', 0))))

    if not recs:
        return []

    # Dédupliquer en conservant le meilleur score par produit
    best = {}
    for prod, score, conf, lift in recs:
        if prod not in best or score > best[prod][0]:
            best[prod] = (score, conf, lift)

    # Trier et formater
    sorted_items = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    result = []
    for prod, (score, conf, lift) in sorted_items[:top_n]:
        result.append({'product': prod, 'score': score, 'confidence': conf, 'lift': lift})

    return result


def build_association_rules(df, method='apriori', min_support=0.01, min_confidence=0.5, max_len=3):
    """
    Construit des règles d'association à partir d'un DataFrame de transactions.

    df : DataFrame avec au minimum les colonnes ['InvoiceNo', 'Description']
    method : 'apriori' ou 'fpgrowth'
    Retourne le DataFrame des règles (columns: antecedents, consequents, support, confidence, lift, ...)
    """
    if not set(["InvoiceNo", "Description"]).issubset(df.columns):
        return pd.DataFrame()

    # Construire la liste des transactions
    transactions = df.groupby("InvoiceNo")["Description"].apply(lambda x: x.dropna().astype(str).tolist()).tolist()
    if not transactions:
        return pd.DataFrame()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_te = pd.DataFrame(te_ary, columns=te.columns_)

    # Filtrer items trop rares
    item_support = df_te.mean()
    frequent_items = item_support[item_support >= min_support].index.tolist()
    if len(frequent_items) == 0:
        return pd.DataFrame()

    df_te_filtered = df_te[frequent_items]

    if method == 'apriori':
        itemsets = apriori(df_te_filtered, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        itemsets = fpgrowth(df_te_filtered, min_support=min_support, use_colnames=True, max_len=max_len)

    try:
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    except Exception:
        # Si pas de règles trouvées
        rules = pd.DataFrame()

    return rules.sort_values(by=["lift", "confidence"], ascending=[False, False]) if not rules.empty else pd.DataFrame()


def compute_rfm(df, snapshot_date=None):
    """
    Calcule la table RFM (Recency, Frequency, Monetary) par CustomerID.
    Retourne un DataFrame RFM avec colonnes [CustomerID, Recency, Frequency, Monetary].
    """
    if not set(["CustomerID", "InvoiceDate", "Total_Prix"]).issubset(df.columns):
        raise ValueError("Les colonnes CustomerID, InvoiceDate et Total_Prix sont requises pour calculer RFM")

    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - pd.to_datetime(x).max()).days,
        "InvoiceNo": "nunique",
        "Total_Prix": "sum"
    }).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    return rfm


def recommend_for_customer(rules, df=None, customer_id=None, user_basket=None, top_n=5,
                           min_confidence=0.1, min_lift=0.0, use_rfm=True):
    """
    Génère des recommandations pour un client en combinant règles d'association et RFM.

    - rules : DataFrame de règles (issue de build_association_rules)
    - df : DataFrame complet des transactions (servira à extraire le panier client et calculer RFM si customer_id fourni)
    - customer_id : identifiant du client pour personnaliser avec RFM
    - user_basket : alternative à customer_id, iterable de produits (Description)
    - top_n : nombre de recommandations retournées
    - use_rfm : si True et si df+customer_id fournis, booste le score selon le montant/fréquence du client

    Retourne une liste de dicts {'product', 'score', 'confidence', 'lift'}
    """
    # Préconditions
    if (rules is None or rules.empty) and df is None:
        return []

    # Si customer_id fourni, extraire panier actuel et calculer RFM
    rfm_row = None
    if customer_id is not None and df is not None:
        # panier client = produits uniques achetés par le client
        try:
            user_basket = df[df['CustomerID'] == customer_id]['Description'].dropna().astype(str).unique().tolist()
        except Exception:
            user_basket = user_basket or []

        try:
            rfm_table = compute_rfm(df)
            rfm_row = rfm_table[rfm_table['CustomerID'] == customer_id]
            if not rfm_row.empty:
                rfm_row = rfm_row.iloc[0]
            else:
                rfm_row = None
        except Exception:
            rfm_row = None

    # Fallback si user_basket non fourni
    if user_basket is None:
        user_basket = []

    # Réutiliser la fonction recommend_products pour filtrer les règles
    base_recs = recommend_products(rules, user_basket, top_n=100, min_confidence=min_confidence, min_lift=min_lift)
    if not base_recs:
        return []

    # Si RFM disponible, calculer un boost
    boost = 1.0
    if use_rfm and rfm_row is not None:
        # boost basé sur Monetary et Frequency (normalisé par max du dataset)
        try:
            rfm_table = compute_rfm(df)
            max_mon = rfm_table['Monetary'].max() if rfm_table['Monetary'].max() > 0 else 1.0
            max_freq = rfm_table['Frequency'].max() if rfm_table['Frequency'].max() > 0 else 1.0
            mon_norm = float(rfm_row['Monetary']) / float(max_mon)
            freq_norm = float(rfm_row['Frequency']) / float(max_freq)
            # pondération simple : 60% Monetary, 40% Frequency
            boost = 1.0 + 0.6 * mon_norm + 0.4 * freq_norm
        except Exception:
            boost = 1.0

    # Appliquer boost aux scores et renvoyer top_n
    recs = []
    for r in base_recs:
        recs.append({'product': r['product'], 'score': r['score'] * boost, 'confidence': r['confidence'], 'lift': r['lift']})

    # Trier et renvoyer
    recs_sorted = sorted(recs, key=lambda x: x['score'], reverse=True)
    return recs_sorted[:top_n]
