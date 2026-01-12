import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")

st.title("🚀 Simulateur de Patrimoine : Monte Carlo & DCA")
st.markdown("Ce simulateur utilise une distribution **log-normale** pour projeter ton épargne.")

# --- SIDEBAR : PARAMÈTRES ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    capital_initial = st.number_input("Capital Initial (€)", value=10000)
    dca_mensuel = st.number_input("Versement mensuel (€)", value=500)
    duree_ans = st.slider("Durée (années)", 1, 40, 20)
    
    st.subheader("Marché")
    rendement_moyen = st.slider("Rendement Annuel Moyen (%)", 0.0, 15.0, 8.5) / 100
    volatilite = st.slider("Volatilité Annuelle (%)", 0.0, 40.0, 15.0) / 100
    inflation = st.slider("Inflation Annuelle (%)", 0.0, 10.0, 2.5) / 100
    
    nb_simulations = st.number_input("Nombre de simulations", value=100, step=50)
    run_button = st.button("Lancer la simulation")

# --- MOTEUR DE CALCUL ---
if run_button:
    months = duree_ans * 12
    # Ajustement des paramètres pour le mensuel (formule de géométrie stochastique)
    mu_mensuel = (1 + rendement_moyen)**(1/12) - 1
    sigma_mensuel = volatilite / np.sqrt(12)
    inflation_mensuelle = (1 + inflation)**(1/12) - 1

    # Matrice pour stocker les résultats
    all_scenarios = np.zeros((months + 1, nb_simulations))
    all_scenarios[0] = capital_initial

    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulation pas à pas pour l'effet visuel
    for m in range(1, months + 1):
        # Tirage log-normal : exp( (mu - sigma^2/2) + sigma * Z )
        rendements_aleatoires = np.random.normal(
            mu_mensuel - 0.5 * sigma_mensuel**2, 
            sigma_mensuel, 
            nb_simulations
        )
        variations = np.exp(rendements_aleatoires)
        
        # Calcul du nouveau solde : (Précédent * Variation) + DCA
        all_scenarios[m] = (all_scenarios[m-1] * variations) + dca_mensuel
        
        if m % 12 == 0:
            progress_bar.progress(m / months)
            status_text.text(f"Calcul de l'année {m//12}...")

    # --- AFFICHAGE GRAPHIQUE ---
    df_scenarios = pd.DataFrame(all_scenarios)
    
    fig = go.Figure()
    # On affiche tous les scénarios avec une opacité faible
    for i in range(nb_simulations):
        fig.add_trace(go.Scatter(y=df_scenarios[i], mode='lines', 
                                 line=dict(width=1), opacity=0.1, 
                                 showlegend=False, name=f"Simu {i}"))

    # Ajout des percentiles (Médiane, 10th, 90th)
    median_path = np.percentile(all_scenarios, 50, axis=1)
    low_path = np.percentile(all_scenarios, 10, axis=1)
    high_path = np.percentile(all_scenarios, 90, axis=1)

    fig.add_trace(go.Scatter(y=median_path, line=dict(color='yellow', width=3), name="Médiane (50%)"))
    fig.add_trace(go.Scatter(y=low_path, line=dict(color='red', width=2, dash='dash'), name="Pessimiste (10%)"))
    fig.add_trace(go.Scatter(y=high_path, line=dict(color='green', width=2, dash='dash'), name="Optimiste (90%)"))

    fig.update_layout(title="Évolution du capital (Nominal)", xaxis_title="Mois", yaxis_title="Euros (€)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- STATS FINALES ---
    final_nominal = all_scenarios[-1]
    final_reel = final_nominal / ((1 + inflation)**duree_ans)

    col1, col2, col3 = st.columns(3)
    col1.metric("Médiane (Réel)", f"{int(np.median(final_reel)):,} €".replace(',', ' '))
    col2.metric("Pessimiste 10% (Réel)", f"{int(np.percentile(final_reel, 10)):,} €".replace(',', ' '))
    col3.metric("Optimiste 90% (Réel)", f"{int(np.percentile(final_reel, 90)):,} €".replace(',', ' '))

    st.success("Simulation terminée ! Les montants affichés en bas sont ajustés à l'inflation.")
