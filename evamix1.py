import streamlit as st
import numpy as np
import pandas as pd

class Critere:
    def __init__(self, min_max: int, poids: float, quant_qual: int):
        self.min_max = min_max
        self.poids = poids
        self.quant_qual = quant_qual

def calculDominances(P, criteres, c):
    """Calcul des matrices alpha et mu (dominances partielles)"""
    nb_alternatives, nb_criteres = P.shape
    alpha = np.zeros((nb_alternatives, nb_alternatives))
    mu = np.zeros((nb_alternatives, nb_alternatives))

    for i in range(nb_alternatives):
        for j in range(nb_alternatives):
            if i != j:
                for k in range(nb_criteres):
                    diff = P[i, k] - P[j, k]
                    poids_c = criteres[k].poids ** c

                    if criteres[k].quant_qual < 0.5:  # Critère qualitatif
                        if diff > 0:
                            alpha[i, j] += poids_c
                        elif diff < 0:
                            alpha[i, j] -= poids_c
                    else:  # Critère quantitatif
                        if diff > 0:
                            mu[i, j] += poids_c
                        elif diff < 0:
                            mu[i, j] -= poids_c

                alpha[i, j] = alpha[i, j] ** (1/c)
                mu[i, j] = mu[i, j] ** (1/c)

    return alpha, mu

def normalisationDominances(alpha, mu):
    """Normalisation des scores de dominance"""
    d1 = np.zeros_like(alpha)
    d2 = np.zeros_like(mu)

    max_alpha, min_alpha = np.max(alpha), np.min(alpha)
    max_mu, min_mu = np.max(mu), np.min(mu)

    range_alpha = max_alpha - min_alpha
    range_mu = max_mu - min_mu

    if range_alpha > 0:
        d1 = (alpha - min_alpha) / range_alpha
    if range_mu > 0:
        d2 = (mu - min_mu) / range_mu

    return d1, d2

def calculDominanceGlobale(d1, d2, criteres):
    """Calcul des scores de dominance globaux"""
    nb_alternatives = d1.shape[0]
    d = np.zeros((nb_alternatives, nb_alternatives))

    # Calcul des sommes des poids
    w1 = sum(critere.poids for critere in criteres if critere.quant_qual == 0)
    w2 = sum(critere.poids for critere in criteres if critere.quant_qual == 1)

    d = w1 * d1 + w2 * d2
    return d

def calculScoresFinaux(d):
    """Calcul des scores finaux"""
    nb_alternatives = d.shape[0]
    scores = np.zeros(nb_alternatives)

    for i in range(nb_alternatives):
        somme = sum(d[j, i] / d[i, j] for j in range(nb_alternatives) if i != j and d[j, i] > 0)
        scores[i] = somme ** -1

    return scores

def evamix(P, criteres, c):
    """Fonction principale EVAMIX"""
    # Normalisation initiale de la matrice de performance
    P_norm = P.copy()
    nb_alternatives, nb_criteres = P.shape

    for j in range(nb_criteres):
        min_val, max_val = np.min(P[:, j]), np.max(P[:, j])
        range_val = max_val - min_val

        if range_val > 0:
            for i in range(nb_alternatives):
                if criteres[j].min_max == 1:  # Maximisation
                    P_norm[i, j] = (P[i, j] - min_val) / range_val
                else:  # Minimisation
                    P_norm[i, j] = (max_val - P[i, j]) / range_val

    # Calcul des dominances
    alpha, mu = calculDominances(P_norm, criteres, c)

    # Normalisation des dominances
    d1, d2 = normalisationDominances(alpha, mu)

    # Calcul des scores de dominance globaux
    d_global = calculDominanceGlobale(d1, d2, criteres)

    # Calcul des scores finaux
    scoresFinaux = calculScoresFinaux(d_global)

    # Calcul l'ordre finale des scores finaux
    ordrefinaux = np.argsort(scoresFinaux)[::-1]

    return ordrefinaux, scoresFinaux

    
def main():
    st.set_page_config(page_title="EVAMIX Multi-Criteria Analysis", page_icon="📊", layout="wide")
    
    # Configuration de style avec une couleur de fond différente et des titres plus visibles
    st.markdown("""
    <style>
    .stApp {
        background-color: #E6E6FA;  /* Lavender Light */
    }
    h1, h2, h3, .stTitle, .stHeader, .stSubheader {
        color: #4A4A8A !important;  /* Couleur de titre plus foncée et visible */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1) !important;
    }
    .stDataFrame {
        max-width: 100%;
        overflow-x: auto;
    }
    .stButton>button {
        color: white !important;
        background-color: #6A5ACD !important;  /* Slate Blue */
        border-radius: 10px;
    }
    /* Style for C Parameter label to match main headers */
    .c-parameter-label {
        color: #4A4A8A !important;
        font-weight: bold;
        font-size: 1.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 EVAMIX Multi-Criteria Decision Analysis")

    # Sidebar pour les inputs
    st.sidebar.header("Configuration")
    num_alternatives = st.sidebar.number_input("Nombre d'alternatives", min_value=1, max_value=20, value=5)
    num_criteres = st.sidebar.number_input("Nombre de critères", min_value=1, max_value=20, value=5)

    # Matrice de Performance
    st.header("Matrice de Performance")
    
    # Initialisation de la matrice de performance
    performance_df = pd.DataFrame(
        np.zeros((num_alternatives, num_criteres)), 
        columns=[f'Critère {j+1}' for j in range(num_criteres)],
        index=[f'Alternative {i+1}' for i in range(num_alternatives)]
    )
    
    # Editeur de matrice de performance
    edited_performance_df = st.data_editor(
        performance_df, 
        num_rows="fixed"
    )

    # Configuration des Critères
    st.header("Configuration des Critères")
    
    # Types de critères séparés avec des listes déroulantes
    criteres_df = pd.DataFrame({
        'Type Max/Min': ['Max'] * num_criteres,
        'Type Quant/Qual': ['Quant'] * num_criteres,
        'Poids': [1/num_criteres] * num_criteres
    }, index=[f'Critère {j+1}' for j in range(num_criteres)])
    
    # Editeur de configuration des critères avec des listes déroulantes
    edited_criteres_df = st.data_editor(
        criteres_df, 
        num_rows="fixed",
        column_config={
            "Type Max/Min": st.column_config.SelectboxColumn(
                "Type Max/Min",
                options=["Max", "Min"],
                required=True
            ),
            "Type Quant/Qual": st.column_config.SelectboxColumn(
                "Type Quant/Qual",
                options=["Quant", "Qual"],
                required=True
            )
        }
    )

    # Conversion des critères
    criteres_inputs = []
    for j, row in edited_criteres_df.iterrows():
        min_max = 1 if row['Type Max/Min'] == 'Max' else 0
        quant_qual = 1 if row['Type Quant/Qual'] == 'Quant' else 0
        criteres_inputs.append(Critere(min_max, row['Poids'], quant_qual))

    # Paramètres Avancés
    st.markdown('<div class="c-parameter-label">Paramètre C</div>', unsafe_allow_html=True)
    c_param = st.number_input("", min_value=1, max_value=10, value=1)

    # Bouton d'analyse
    if st.button("Lancer l'Analyse EVAMIX"):
        P = edited_performance_df.values
        ordre, scores = evamix(P, criteres_inputs, c_param)
        
        results_df = pd.DataFrame({
            'Alternative': [f"Alternative {x+1}" for x in ordre],
            'Score': scores[ordre]
        })
        
        st.header("Résultats de l'Analyse")
        st.dataframe(results_df.style.highlight_max(subset=['Score'], color='lightgreen'), use_container_width=True)

if __name__ == "__main__":
    main()