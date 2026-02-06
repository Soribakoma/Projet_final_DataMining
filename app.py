import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import load_data, prepare_kmeans_data
from utils import load_uploaded_file
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import io
import base64

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Data Mining E-commerce Platform",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded",
)

# ==================== CSS MODERNE ====================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-color: #0f4c81;
        --secondary-color: #2c5f8d;
        --accent-color: #4a90e2;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --bg-primary: #f8f9fa;
        --bg-secondary: #ffffff;
        --text-primary: #2c3e50;
        --text-secondary: #7f8c8d;
        --border-color: #e0e6ed;
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header Styles */
    .app-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .app-logo {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Card Styles */
    .modern-card {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .card-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-color);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
        border-color: var(--accent-color);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-color);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--accent-color);
        display: inline-block;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success-color) 0%, #27ae60 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-secondary);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-color);
        background: #f8f9fa;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f8f9fa;
        border-color: var(--accent-color);
    }
    
    /* DataFrame Styles */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--accent-color);
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border-color);
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .footer-content {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .footer-link {
        color: var(--accent-color);
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .footer-link:hover {
        color: var(--primary-color);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-color) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 8px;
        border-color: var(--border-color);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background-color: var(--accent-color);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .app-header {
            padding: 1.5rem 1rem;
        }
        
        .app-logo {
            font-size: 1.8rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== FONCTIONS UTILITAIRES ====================

def load_uploaded_file(file):
    """Charge un fichier CSV avec gestion d'encodage"""
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="latin1")

def compute_basic_metrics(df):
    """Calcule les métriques de base du dataset"""
    n_rows = len(df)
    n_customers = df['CustomerID'].nunique() if 'CustomerID' in df.columns else None
    n_products = df['Description'].nunique() if 'Description' in df.columns else None
    n_transactions = df['InvoiceNo'].nunique() if 'InvoiceNo' in df.columns else None
    return n_rows, n_customers, n_products, n_transactions

@st.cache_data
def build_transactions_from_df(df):
    """Construit la liste des transactions pour les règles d'association"""
    if not set(["InvoiceNo", "Description"]).issubset(df.columns):
        return []
    transactions = df.groupby("InvoiceNo")["Description"].apply(
        lambda x: x.dropna().astype(str).tolist()
    ).tolist()
    return transactions

def df_to_bytes_download(df):
    """Convertit un DataFrame en bytes pour le téléchargement"""
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite

def create_metric_card(icon, value, label):
    """Crée une carte métrique moderne"""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def qcut_safe(series, q=4):
    """Applique pd.qcut en gérant les doublons"""
    try:
        return pd.qcut(series, q, labels=list(range(1, q+1)), duplicates='drop')
    except ValueError:
        bins = pd.qcut(series, q, duplicates='drop').cat.categories
        n_bins = len(bins)
        return pd.qcut(series, q, labels=list(range(1, n_bins+1)), duplicates='drop')

def recommend_products(rules, user_basket, top_n=5, min_confidence=0.7, min_lift=1.5):
    """
    Recommande des produits basés sur les règles d'association
    
    Args:
        rules: DataFrame des règles Apriori/FP-Growth
        user_basket: set de produits saisis par l'utilisateur (en minuscules)
        top_n: nombre maximum de recommandations
        min_confidence: confiance minimale requise
        min_lift: lift minimum requis
    
    Returns:
        list: liste des produits recommandés
    """
    # Filtrer les règles pertinentes
    filtered_rules = rules[
        (rules['confidence'] >= min_confidence) &
        (rules['lift'] >= min_lift)
    ]
    
    recommendations = []
    recommendation_scores = {}
    
    for _, row in filtered_rules.iterrows():
        antecedents = set(str(item).lower() for item in row['antecedents'])
        
        # Vérifier si les antécédents sont inclus dans le panier utilisateur
        if antecedents.issubset(user_basket):
            consequents = set(str(item).lower() for item in row['consequents'])
            
            # Calculer un score pour chaque recommandation
            for item in consequents:
                if item not in user_basket:
                    score = row['confidence'] * row['lift']
                    if item in recommendation_scores:
                        recommendation_scores[item] = max(recommendation_scores[item], score)
                    else:
                        recommendation_scores[item] = score
    
    # Trier par score et retourner top_n
    sorted_recommendations = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [(item, score) for item, score in sorted_recommendations[:top_n]]

# ==================== HEADER ====================
st.markdown("""
<div class="app-header">
    <div class="app-logo">🛍️ Data Mining E-commerce Platform</div>
    <div class="app-subtitle">Analyse avancée • APRIORI • FP-GROWTH • K-means • RFM</div>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.markdown("## 📊 Navigation")
st.sidebar.markdown("---")

menu = st.sidebar.selectbox(
    "🔍 Choisir une analyse",
    [
        "📈 Analyse descriptive",
        "📦 Segmentation K-means",
        "💰 Segmentation RFM",
        "🧾 APRIORI",
        "🌲 FP-GROWTH",
        "🎯 Recommandations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎨 Personnalisation")

theme_choice = st.sidebar.selectbox(
    "Thème de couleur",
    ["Bleu (par défaut)", "Vert", "Orange", "Pourpre", "Gris"]
)

theme_map = {
    "Bleu (par défaut)": "#0f4c81",
    "Vert": "#2a9d8f",
    "Orange": "#f4a261",
    "Pourpre": "#6a4c93",
    "Gris": "#495057",
}
primary_color = theme_map.get(theme_choice, "#0f4c81")

# Configuration Seaborn
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['text.color'] = primary_color
plt.rcParams['axes.labelcolor'] = primary_color

# ==================== PAGE: ANALYSE DESCRIPTIVE ====================
if menu == "📈 Analyse descriptive":
    st.markdown('<p class="section-header">📈 Analyse Exploratoire des Données</p>', unsafe_allow_html=True)
    
    file = st.file_uploader(
        "📂 Charger votre base de données",
        type=["csv", "xlsx"],
        help="Formats acceptés: CSV, XLSX"
    )
    
    if file is not None:
        try:
            if file.name.endswith(".csv"):
                df = load_uploaded_file(file)
            else:
                df = pd.read_excel(file)
            
            st.success("✅ Fichier chargé avec succès!")
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {e}")
            st.stop()
        
        # Métriques
        n_rows, n_customers, n_products, n_transactions = compute_basic_metrics(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("📊", f"{n_rows:,}", "Lignes totales"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("👥", f"{n_customers:,}", "Clients uniques"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("🏷️", f"{n_products:,}", "Produits distincts"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("🧾", f"{n_transactions:,}", "Transactions"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Aperçu des données
        st.markdown('<p class="section-header">📋 Aperçu des Données</p>', unsafe_allow_html=True)
        
        with st.expander("🔍 Afficher les premières lignes"):
            st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("📊 Statistiques descriptives"):
            st.dataframe(df.describe(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Télécharger le dataset (CSV)",
                data=df_to_bytes_download(df),
                file_name="data_export.csv",
                mime="text/csv"
            )
        
        # Visualisations
        st.markdown('<p class="section-header">📊 Visualisations</p>', unsafe_allow_html=True)
        
        if "Total_Prix" in df.columns:
            st.markdown("#### 💵 Distribution des montants")
            fig = px.histogram(
                df,
                x="Total_Prix",
                nbins=50,
                title="Distribution des montants des transactions (échelle log)",
                labels={"Total_Prix": "Montant (€)", "count": "Fréquence"}
            )
            fig.update_traces(marker_color=primary_color, marker_line_color='white', marker_line_width=1)
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=primary_color),
                yaxis_type='log',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if "Description" in df.columns and "Quantity" in df.columns:
            st.markdown("#### 🏆 Top 10 des produits les plus vendus")
            top_products = (
                df.groupby("Description")["Quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            
            fig = px.bar(
                x=top_products.values,
                y=top_products.index,
                orientation='h',
                labels={'x': 'Quantité vendue', 'y': 'Produit'},
                title="Top 10 des produits"
            )
            fig.update_traces(marker_color=primary_color, marker_line_color='white', marker_line_width=1)
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=primary_color),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: K-MEANS ====================
elif menu == "📦 Segmentation K-means":
    st.markdown('<p class="section-header">📦 Segmentation Client avec K-means</p>', unsafe_allow_html=True)
    
    file = st.file_uploader(
        "📂 Charger la base de données",
        type=["csv", "xlsx"],
        help="Laissez vide pour utiliser les données par défaut"
    )
    
    if file is not None:
        if file.name.endswith('.csv'):
            df_input = load_uploaded_file(file)
        else:
            df_input = pd.read_excel(file)
    else:
        df_input = load_data("donnees_ecommerce_netoyer.csv")
    
    df_grouped, X_scaled = prepare_kmeans_data(df_input)
    
    # Métriques
    n_rows, n_customers, n_products, n_transactions = compute_basic_metrics(df_input)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(create_metric_card("👥", f"{n_customers:,}", "Clients"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("🧾", f"{n_transactions:,}", "Transactions"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("📊", f"{n_rows:,}", "Lignes"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Paramètres
    col1, col2 = st.columns([2, 1])
    with col1:
        k = st.slider("🎯 Nombre de clusters", 2, 8, 4, help="Sélectionnez le nombre de segments clients")
    
    with col2:
        show_elbow = st.checkbox("📉 Afficher courbe Elbow", help="Méthode du coude pour choisir K optimal")
    
    if show_elbow:
        with st.spinner("📊 Calcul de la courbe Elbow..."):
            inertias = []
            silhouettes = []
            K_range = range(2, 11)
            
            for kk in K_range:
                km = KMeans(n_clusters=kk, random_state=42, n_init=10)
                labels_temp = km.fit_predict(X_scaled)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(X_scaled, labels_temp))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(K_range),
                y=inertias,
                mode='lines+markers',
                name='Inertie',
                line=dict(color=primary_color, width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Méthode du Coude (Elbow Method)",
                xaxis_title="Nombre de clusters (k)",
                yaxis_title="Inertie",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=primary_color)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Application K-means
    with st.spinner("🔄 Calcul des clusters en cours..."):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        df_grouped["Cluster"] = labels
        
        sil_score = silhouette_score(X_scaled, labels)
    
    st.success(f"✅ Segmentation terminée! Score de silhouette: {sil_score:.3f}")
    
    # Résumé des clusters
    st.markdown('<p class="section-header">📊 Analyse des Segments</p>', unsafe_allow_html=True)
    
    summary = df_grouped.groupby("Cluster")[['Frequency', 'Monetary', 'Recency']].mean().round(2)
    summary['Taille'] = df_grouped.groupby("Cluster").size()
    
    st.dataframe(summary.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    st.download_button(
        "⬇️ Télécharger l'analyse des clusters",
        data=df_to_bytes_download(summary),
        file_name="kmeans_clusters_analysis.csv",
        mime="text/csv"
    )
    
    # Visualisation
    st.markdown('<p class="section-header">📈 Visualisation des Clusters</p>', unsafe_allow_html=True)
    
    fig = px.scatter(
        df_grouped,
        x='Monetary',
        y='Frequency',
        color='Cluster',
        size='Recency',
        hover_data=['CustomerID'],
        title='Segmentation Client (Monetary vs Frequency)',
        labels={'Monetary': 'Valeur Monétaire (€)', 'Frequency': 'Fréquence d\'achat'},
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=primary_color)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: RFM ====================
elif menu == "💰 Segmentation RFM":
    st.markdown('<p class="section-header">💰 Analyse RFM (Recency, Frequency, Monetary)</p>', unsafe_allow_html=True)
    
    df = load_data("donnees_ecommerce_netoyer.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Informations sur les dates
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📅 **Date début:** {df['InvoiceDate'].min().strftime('%d/%m/%Y')}")
    with col2:
        st.info(f"📅 **Date fin:** {df['InvoiceDate'].max().strftime('%d/%m/%Y')}")
    with col3:
        duration = (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days
        st.info(f"⏱️ **Durée:** {duration} jours")
    
    # Calcul RFM
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - pd.to_datetime(x).max()).days,
        "InvoiceNo": "nunique",
        "Total_Prix": "sum"
    }).reset_index()
    
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    
    # Scores RFM
    rfm["R_Score"] = qcut_safe(rfm["Recency"], 4)
    rfm["F_Score"] = qcut_safe(rfm["Frequency"], 4)
    rfm["M_Score"] = qcut_safe(rfm["Monetary"], 4)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) +
        rfm["F_Score"].astype(str) +
        rfm["M_Score"].astype(str)
    )
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric_card("👥", f"{rfm['CustomerID'].nunique():,}", "Clients analysés"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("💵", f"{rfm['Monetary'].mean():.2f}€", "Valeur moyenne"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("🔄", f"{rfm['Frequency'].mean():.1f}", "Fréquence moyenne"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("📅", f"{rfm['Recency'].mean():.0f}j", "Récence moyenne"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tableau RFM
    st.markdown('<p class="section-header">📊 Tableau RFM</p>', unsafe_allow_html=True)
    st.dataframe(rfm.head(20), use_container_width=True)
    
    st.download_button(
        "⬇️ Télécharger l'analyse RFM complète",
        data=df_to_bytes_download(rfm),
        file_name="rfm_analysis.csv",
        mime="text/csv"
    )
    
    # Distribution des scores
    st.markdown('<p class="section-header">📈 Distribution des Scores RFM</p>', unsafe_allow_html=True)
    
    score_counts = rfm["RFM_Score"].value_counts().head(20)
    fig = px.bar(
        x=score_counts.values,
        y=score_counts.index,
        orientation='h',
        title="Top 20 des scores RFM",
        labels={'x': 'Nombre de clients', 'y': 'Score RFM'}
    )
    fig.update_traces(marker_color=primary_color)
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=primary_color)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: APRIORI ====================
elif menu == "🧾 APRIORI":
    st.markdown('<p class="section-header">🧾 Règles d\'Association - Algorithme APRIORI</p>', unsafe_allow_html=True)
    
    file = st.file_uploader("📂 Charger la base de données", type=["csv", "xlsx"])
    
    if file is not None:
        if file.name.endswith('.csv'):
            df_rules = load_uploaded_file(file)
        else:
            df_rules = pd.read_excel(file)
    else:
        df_rules = load_data("donnees_ecommerce_netoyer.csv")
    
    if not set(["InvoiceNo", "Description"]).issubset(df_rules.columns):
        st.error("❌ Les colonnes 'InvoiceNo' et 'Description' sont requises!")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Paramètres")
        min_support = st.slider("📊 Support minimum", 0.001, 0.1, 0.01, step=0.001)
        min_conf = st.slider("🎯 Confiance minimale", 0.1, 1.0, 0.5, step=0.05)
        max_len = st.number_input("📏 Taille max itemset", 1, 5, 3)
        
        st.info(f"""
        **Configuration actuelle:**
        - Support: {min_support*100:.2f}%
        - Confiance: {min_conf*100:.0f}%
        - Taille max: {max_len}
        """)
    
    with col2:
        with st.spinner("🔄 Extraction des règles en cours..."):
            transactions = build_transactions_from_df(df_rules)
            
            if not transactions:
                st.warning("⚠️ Aucune transaction valide détectée!")
                st.stop()
            
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_te = pd.DataFrame(te_ary, columns=te.columns_)
            
            item_support = df_te.mean()
            frequent_items = item_support[item_support >= min_support].index.tolist()
            
            if len(frequent_items) == 0:
                st.warning("⚠️ Aucun item fréquent. Réduisez le support minimum.")
                st.stop()
            
            df_te_filtered = df_te[frequent_items]
            
            try:
                frequent_itemsets = apriori(
                    df_te_filtered,
                    min_support=min_support,
                    use_colnames=True,
                    max_len=max_len
                )
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=min_conf
                )
                rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False])
                
                st.success(f"✅ {len(frequent_itemsets)} itemsets et {len(rules)} règles trouvés!")
                
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                st.stop()
        
        # Résultats
        st.markdown("### 📊 Itemsets Fréquents (Top 20)")
        st.dataframe(
            frequent_itemsets.sort_values(by='support', ascending=False).head(20),
            use_container_width=True
        )
        
        st.markdown("### 📋 Règles d'Association (Top 50)")
        if not rules.empty:
            rules_display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(lambda s: ", ".join(list(s)))
            rules_display['consequents'] = rules_display['consequents'].apply(lambda s: ", ".join(list(s)))
            st.dataframe(rules_display.head(50), use_container_width=True)
            
            # Visualisation
            fig = px.scatter(
                rules_display.head(30),
                x='support',
                y='confidence',
                size='lift',
                hover_data=['antecedents', 'consequents'],
                title='Visualisation des Règles (Support vs Confidence)',
                labels={'support': 'Support', 'confidence': 'Confiance', 'lift': 'Lift'}
            )
            fig.update_traces(marker=dict(color=primary_color, line=dict(width=1, color='white')))
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color=primary_color))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Aucune règle trouvée. Ajustez les paramètres.")
        
        # Téléchargements
        col_a, col_b = st.columns(2)
        with col_a:
            if not frequent_itemsets.empty:
                st.download_button(
                    "⬇️ Itemsets (CSV)",
                    data=df_to_bytes_download(frequent_itemsets),
                    file_name="apriori_itemsets.csv"
                )
        with col_b:
            if not rules.empty:
                st.download_button(
                    "⬇️ Règles (CSV)",
                    data=df_to_bytes_download(rules),
                    file_name="apriori_rules.csv"
                )

# ==================== PAGE: FP-GROWTH ====================
elif menu == "🌲 FP-GROWTH":
    st.markdown('<p class="section-header">🌲 Règles d\'Association - Algorithme FP-Growth</p>', unsafe_allow_html=True)
    
    file = st.file_uploader("📂 Charger la base de données", type=["csv", "xlsx"], key='fp')
    
    if file is not None:
        if file.name.endswith('.csv'):
            df_rules = load_uploaded_file(file)
        else:
            df_rules = pd.read_excel(file)
    else:
        df_rules = load_data("donnees_ecommerce_netoyer.csv")
    
    if not set(["InvoiceNo", "Description"]).issubset(df_rules.columns):
        st.error("❌ Les colonnes 'InvoiceNo' et 'Description' sont requises!")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Paramètres")
        min_support = st.slider("📊 Support minimum", 0.001, 0.1, 0.01, step=0.001, key='fp_supp')
        min_conf = st.slider("🎯 Confiance minimale", 0.1, 1.0, 0.5, step=0.05, key='fp_conf')
        max_len = st.number_input("📏 Taille max itemset", 1, 5, 3, key='fp_max')
        
        st.info(f"""
        **Configuration actuelle:**
        - Support: {min_support*100:.2f}%
        - Confiance: {min_conf*100:.0f}%
        - Taille max: {max_len}
        
        🚀 FP-Growth est généralement plus rapide qu'APRIORI
        """)
    
    with col2:
        with st.spinner("🔄 Extraction FP-Growth en cours..."):
            transactions = build_transactions_from_df(df_rules)
            
            if not transactions:
                st.warning("⚠️ Aucune transaction valide détectée!")
                st.stop()
            
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_te = pd.DataFrame(te_ary, columns=te.columns_)
            
            item_support = df_te.mean()
            frequent_items = item_support[item_support >= min_support].index.tolist()
            
            if len(frequent_items) == 0:
                st.warning("⚠️ Aucun item fréquent. Réduisez le support minimum.")
                st.stop()
            
            df_te_filtered = df_te[frequent_items]
            
            try:
                frequent_itemsets = fpgrowth(
                    df_te_filtered,
                    min_support=min_support,
                    use_colnames=True,
                    max_len=max_len
                )
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=min_conf
                )
                rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False])
                
                st.success(f"✅ {len(frequent_itemsets)} itemsets et {len(rules)} règles trouvés!")
                
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                st.stop()
        
        # Résultats
        st.markdown("### 📊 Itemsets Fréquents (Top 20)")
        st.dataframe(
            frequent_itemsets.sort_values(by='support', ascending=False).head(20),
            use_container_width=True
        )
        
        st.markdown("### 📋 Règles d'Association (Top 50)")
        if not rules.empty:
            rules_display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(lambda s: ", ".join(list(s)))
            rules_display['consequents'] = rules_display['consequents'].apply(lambda s: ", ".join(list(s)))
            st.dataframe(rules_display.head(50), use_container_width=True)
            
            # Visualisation
            fig = px.scatter(
                rules_display.head(30),
                x='support',
                y='confidence',
                size='lift',
                color='lift',
                hover_data=['antecedents', 'consequents'],
                title='Visualisation des Règles FP-Growth',
                labels={'support': 'Support', 'confidence': 'Confiance', 'lift': 'Lift'},
                color_continuous_scale='Viridis'
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color=primary_color))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Aucune règle trouvée. Ajustez les paramètres.")
        
        # Téléchargements
        col_a, col_b = st.columns(2)
        with col_a:
            if not frequent_itemsets.empty:
                st.download_button(
                    "⬇️ Itemsets (CSV)",
                    data=df_to_bytes_download(frequent_itemsets),
                    file_name="fpgrowth_itemsets.csv"
                )
        with col_b:
            if not rules.empty:
                st.download_button(
                    "⬇️ Règles (CSV)",
                    data=df_to_bytes_download(rules),
                    file_name="fpgrowth_rules.csv"
                )

# ==================== PAGE: RECOMMANDATIONS ====================
elif menu == "🎯 Recommandations":
    st.markdown('<p class="section-header">🎯 Système de Recommandation de Produits</p>', unsafe_allow_html=True)
    
    st.info("""
    💡 **Comment ça marche ?**
    
    Ce système utilise les règles d'association (APRIORI ou FP-Growth) pour recommander des produits 
    en fonction du panier de l'utilisateur. Plus la confiance et le lift sont élevés, plus la recommandation est pertinente.
    """)
    
    # Choix de l'algorithme
    col1, col2 = st.columns(2)
    with col1:
        algo_choice = st.radio(
            "🔧 Algorithme à utiliser",
            ["APRIORI", "FP-GROWTH"],
            horizontal=True
        )
    
    # Upload ou données par défaut
    file = st.file_uploader(
        "📂 Charger la base de données (optionnel)",
        type=["csv", "xlsx"],
        key='reco_file'
    )
    
    if file is not None:
        if file.name.endswith('.csv'):
            df_rules = load_uploaded_file(file)
        else:
            df_rules = pd.read_excel(file)
    else:
        df_rules = load_data("donnees_ecommerce_netoyer.csv")
    
    if not set(["InvoiceNo", "Description"]).issubset(df_rules.columns):
        st.error("❌ Les colonnes 'InvoiceNo' et 'Description' sont requises!")
        st.stop()
    
    # Paramètres dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Paramètres de Recommandation")
    min_support_reco = st.sidebar.slider("📊 Support minimum", 0.001, 0.1, 0.01, step=0.001, key='reco_supp')
    min_conf_reco = st.sidebar.slider("🎯 Confiance minimale", 0.1, 1.0, 0.7, step=0.05, key='reco_conf')
    min_lift_reco = st.sidebar.slider("📈 Lift minimum", 1.0, 5.0, 1.5, step=0.1, key='reco_lift')
    top_n = st.sidebar.number_input("🔢 Nombre de recommandations", 1, 20, 5, key='reco_topn')
    
    # Extraction des règles
    with st.spinner(f"🔄 Extraction des règles avec {algo_choice}..."):
        transactions = build_transactions_from_df(df_rules)
        
        if not transactions:
            st.warning("⚠️ Aucune transaction valide détectée!")
            st.stop()
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_te = pd.DataFrame(te_ary, columns=te.columns_)
        
        item_support = df_te.mean()
        frequent_items = item_support[item_support >= min_support_reco].index.tolist()
        
        if len(frequent_items) == 0:
            st.warning("⚠️ Aucun item fréquent. Réduisez le support minimum.")
            st.stop()
        
        df_te_filtered = df_te[frequent_items]
        
        try:
            if algo_choice == "APRIORI":
                frequent_itemsets = apriori(
                    df_te_filtered,
                    min_support=min_support_reco,
                    use_colnames=True,
                    max_len=4
                )
            else:
                frequent_itemsets = fpgrowth(
                    df_te_filtered,
                    min_support=min_support_reco,
                    use_colnames=True,
                    max_len=4
                )
            
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_conf_reco
            )
            rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False])
            
            st.success(f"✅ {len(rules)} règles extraites avec succès!")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'extraction: {e}")
            st.stop()
    
    # Interface de saisie du panier
    st.markdown("---")
    st.markdown('<p class="section-header">🛒 Simuler un Panier Client</p>', unsafe_allow_html=True)
    
    # Obtenir la liste de tous les produits disponibles
    all_products = sorted(df_rules['Description'].dropna().unique())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Méthode 1: Sélection multiple
        st.markdown("#### Méthode 1: Sélection depuis la liste")
        selected_products = st.multiselect(
            "🔍 Sélectionnez des produits",
            options=all_products,
            help="Choisissez un ou plusieurs produits"
        )
        
        # Méthode 2: Saisie manuelle
        st.markdown("#### Méthode 2: Saisie manuelle")
        manual_input = st.text_area(
            "✍️ Ou saisissez des produits (un par ligne)",
            height=100,
            placeholder="white hanging heart t-light holder\nregency cakestand 3 tier\n..."
        )
        
        # Combiner les deux méthodes
        user_basket = set()
        
        if selected_products:
            user_basket.update([p.lower() for p in selected_products])
        
        if manual_input:
            manual_products = [p.strip().lower() for p in manual_input.split('\n') if p.strip()]
            user_basket.update(manual_products)
    
    with col2:
        st.markdown("#### 🛒 Panier Actuel")
        if user_basket:
            st.success(f"**{len(user_basket)} produit(s)**")
            for idx, item in enumerate(sorted(user_basket), 1):
                st.write(f"{idx}. {item.title()}")
        else:
            st.info("Panier vide")
    
    # Bouton de recommandation
    if st.button("🚀 Obtenir des Recommandations", type="primary", use_container_width=True):
        if not user_basket:
            st.warning("⚠️ Veuillez ajouter au moins un produit dans le panier!")
        else:
            with st.spinner("🔮 Génération des recommandations..."):
                recommendations = recommend_products(
                    rules,
                    user_basket,
                    top_n=top_n,
                    min_confidence=min_conf_reco,
                    min_lift=min_lift_reco
                )
                
                if recommendations:
                    st.markdown("---")
                    st.markdown('<p class="section-header">✨ Produits Recommandés</p>', unsafe_allow_html=True)
                    
                    # Affichage des recommandations en colonnes
                    num_cols = min(3, len(recommendations))
                    cols = st.columns(num_cols)
                    
                    for idx, (product, score) in enumerate(recommendations):
                        col_idx = idx % num_cols
                        with cols[col_idx]:
                            st.markdown(f"""
                            <div class="modern-card" style="text-align: center; min-height: 150px;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎁</div>
                                <div style="font-weight: 600; color: {primary_color}; margin-bottom: 0.5rem;">
                                    {product.title()}
                                </div>
                                <div style="font-size: 0.9rem; color: #7f8c8d;">
                                    Score: {score:.2f}
                                </div>
                                <div style="margin-top: 0.5rem;">
                                    <span style="background: {primary_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem;">
                                        #{idx + 1}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Statistiques des recommandations
                    st.markdown("---")
                    st.markdown("### 📊 Détails des Recommandations")
                    
                    reco_df = pd.DataFrame(recommendations, columns=['Produit', 'Score'])
                    reco_df['Rang'] = range(1, len(reco_df) + 1)
                    reco_df = reco_df[['Rang', 'Produit', 'Score']]
                    
                    st.dataframe(reco_df, use_container_width=True)
                    
                    # Graphique des scores
                    fig = px.bar(
                        reco_df,
                        x='Score',
                        y='Produit',
                        orientation='h',
                        title='Scores de Recommandation',
                        labels={'Score': 'Score de Pertinence', 'Produit': 'Produit Recommandé'}
                    )
                    fig.update_traces(marker_color=primary_color)
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color=primary_color),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export
                    st.download_button(
                        "⬇️ Télécharger les recommandations (CSV)",
                        data=df_to_bytes_download(reco_df),
                        file_name="recommandations.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("""
                    ⚠️ **Aucune recommandation trouvée** 
                    
                    Essayez de:
                    - Réduire la confiance minimale
                    - Réduire le lift minimum
                    - Ajouter d'autres produits dans le panier
                    - Réduire le support minimum
                    """)
    
    # Afficher quelques statistiques sur les règles
    st.markdown("---")
    st.markdown("### 📈 Statistiques des Règles")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre de règles", len(rules))
    with col2:
        st.metric("Confiance moyenne", f"{rules['confidence'].mean():.2%}")
    with col3:
        st.metric("Lift moyen", f"{rules['lift'].mean():.2f}")
    
    # Top 10 des règles
    with st.expander("🔍 Voir les meilleures règles d'association"):
        top_rules = rules.head(10).copy()
        top_rules['antecedents'] = top_rules['antecedents'].apply(lambda s: ", ".join(list(s)))
        top_rules['consequents'] = top_rules['consequents'].apply(lambda s: ", ".join(list(s)))
        st.dataframe(
            top_rules[['antecedents', 'consequents', 'confidence', 'lift']],
            use_container_width=True
        )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div class="app-footer">
    <div class="footer-content">
        <p><strong>🎓 Projet Data Mining E-commerce</strong></p>
        <p>Développé par le Groupe 2 • M2 SID • Université Alioune Diop de Bambey</p>
        <p style="margin-top: 1rem; font-size: 0.85rem; color: #95a5a6;">
            © 2026 • Tous droits réservés a la M2 SID • Conformément aux exigences académiques du projet de data mining
        </p>
    </div>
</div>
""", unsafe_allow_html=True)