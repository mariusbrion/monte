import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page Streamlit
st.set_page_config(page_title="Simulateur de Patrimoine Monte Carlo", layout="wide")

def calculate_monte_carlo(initial_cap, monthly_dca, years, mu, sigma, inflation, n_sims=1000):
    """
    Calcule les trajectoires de patrimoine en utilisant une simulation de Monte Carlo.
    Utilise un modèle de mouvement brownien géométrique pour les rendements.
    """
    months = years * 12
    dt = 1/12  # Pas de temps mensuel
    
    # Conversion des paramètres annuels en mensuels
    # Pour un rendement log-normal : ln(1+r)
    mu_monthly = (mu - 0.5 * sigma**2) * dt
    sigma_monthly = sigma * np.sqrt(dt)
    inflation_monthly = (1 + inflation)**(1/12) - 1

    # Initialisation des matrices (Lignes: mois, Colonnes: simulations)
    # On commence au mois 0 avec le capital initial
    trajectories = np.zeros((months + 1, n_sims))
    trajectories_real = np.zeros((months + 1, n_sims))
    trajectories[0] = initial_cap
    trajectories_real[0] = initial_cap

    # Simulation mois par mois
    for t in range(1, months + 1):
        # Génération de rendements aléatoires normaux pour toutes les simulations
        random_returns = np.exp(np.random.normal(mu_monthly, sigma_monthly, n_sims))
        
        # Calcul du capital nominal : (Capital Précédent + Versement) * Rendement
        trajectories[t] = (trajectories[t-1] + monthly_dca) * random_returns
        
        # Calcul du capital réel (ajusté de l'inflation cumulée)
        inflation_factor = (1 + inflation)**(t/12)
        trajectories_real[t] = trajectories[t] / inflation_factor

    return trajectories, trajectories_real

def calculate_max_drawdown(trajectories):
    """Calcule le Maximum Drawdown moyen sur l'ensemble des simulations."""
    drawdowns = []
    for i in range(trajectories.shape[1]):
        series = pd.Series(trajectories[:, i])
        roll_max = series.cummax()
        dd = (series - roll_max) / roll_max
        drawdowns.append(dd.min())
    return np.mean(drawdowns)

# --- INTERFACE UTILISATEUR (SIDEBAR) ---
st.sidebar.header("📈 Paramètres d'investissement")

cap_initial = st.sidebar.number_input("Capital Initial (€)", value=10000, step=1000)
dca_mensuel = st.sidebar.number_input("Versement mensuel (DCA) (€)", value=500, step=50)
duree_ans = st.sidebar.slider("Durée de l'investissement (ans)", 1, 40, 20)

st.sidebar.subheader("Configuration Marché")
rendement_moyen = st.sidebar.slider("Rendement annuel moyen μ (%)", 0.0, 15.0, 7.0) / 100
volatilite = st.sidebar.slider("Volatilité annuelle σ (%)", 0.0, 30.0, 15.0) / 100
tx_inflation = st.sidebar.slider("Inflation annuelle estimée (%)", 0.0, 10.0, 2.0) / 100

# --- CALCULS ---
if st.sidebar.button("Lancer la simulation"):
    with st.spinner('Simulation en cours...'):
        nom_traj, real_traj = calculate_monte_carlo(
            cap_initial, dca_mensuel, duree_ans, rendement_moyen, volatilite, tx_inflation
        )
        
        # --- RÉSUMÉ DES RÉSULTATS ---
        st.title("📊 Résultats de la Simulation Monte Carlo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        final_nom = nom_traj[-1, :]
        final_real = real_traj[-1, :]
        
        with col1:
            st.metric("Médiane (Nominal)", f"{np.median(final_nom):,.0f} €".replace(",", " "))
        with col2:
            st.metric("Médiane (Réel)", f"{np.median(final_real):,.0f} €".replace(",", " "))
        with col3:
            total_investi = cap_initial + (dca_mensuel * duree_ans * 12)
            st.metric("Total Investi", f"{total_investi:,.0f} €".replace(",", " "))
        with col4:
            mdd = calculate_max_drawdown(nom_traj)
            st.metric("Max Drawdown Moyen", f"{mdd:.2%}")

        # --- GRAPHIQUE DES TRAJECTOIRES ---
        st.subheader("📈 Évolution du patrimoine (100 premiers scénarios)")
        
        fig_lines = go.Figure()
        x_axis = np.arange(duree_ans * 12 + 1) / 12
        
        # On affiche seulement les 100 premières simulations pour la lisibilité
        for i in range(min(100, nom_traj.shape[1])):
            fig_lines.add_trace(go.Scatter(
                x=x_axis, y=nom_traj[:, i],
                mode='lines',
                line=dict(width=1),
                opacity=0.3,
                showlegend=False
            ))
            
        # Ajout de la médiane en gras
        fig_lines.add_trace(go.Scatter(
            x=x_axis, y=np.median(nom_traj, axis=1),
            mode='lines',
            name='Médiane',
            line=dict(color='white', width=4)
        ))
        
        fig_lines.update_layout(
            xaxis_title="Années",
            yaxis_title="Capital (€)",
            template="plotly_dark",
            hovermode="x"
        )
        st.plotly_chart(fig_lines, use_container_width=True)

        # --- DISTRIBUTION ET PERCENTILES ---
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("🎯 Distribution du capital final")
            fig_hist = px.histogram(
                pd.DataFrame({'Capital Final (Réel)': final_real}),
                x='Capital Final (Réel)',
                nbins=50,
                title="Probabilité des montants finaux (ajustés inflation)",
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_right:
            st.subheader("📋 Percentiles")
            
            percentiles = [10, 25, 50, 75, 90]
            data_p = {
                "Percentile": [f"{p}th" for p in percentiles],
                "Nominal (€)": [f"{np.percentile(final_nom, p):,.0f}".replace(",", " ") for p in percentiles],
                "Réel (€)": [f"{np.percentile(final_real, p):,.0f}".replace(",", " ") for p in percentiles]
            }
            df_p = pd.DataFrame(data_p)
            st.table(df_p)
            
            st.info("""
            **Interprétation :**
            - **10th percentile :** Il y a 90% de chances que votre capital soit supérieur à ce montant (scénario pessimiste).
            - **50th percentile :** La valeur médiane (scénario probable).
            - **90th percentile :** Il y a 10% de chances d'atteindre ou dépasser ce montant (scénario optimiste).
            """)

else:
    # Message d'accueil avant le lancement
    st.title("Simulateur de Patrimoine Monte Carlo")
    st.markdown("""
    Bienvenue dans cet outil d'aide à la décision financière. 
    
    ### Comment ça marche ?
    1. Ajustez vos paramètres dans la barre latérale à gauche.
    2. Le simulateur génère **1 000 scénarios** basés sur une distribution log-normale des rendements.
    3. Les résultats tiennent compte de la capitalisation composée, de vos versements mensuels et de l'inflation.
    
    Cliquez sur **'Lancer la simulation'** pour voir les projections.
    """)
    
    st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?auto=format&fit=crop&q=80&w=1000", caption="Analyse de données financières")
