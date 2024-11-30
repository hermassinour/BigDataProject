import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import base64
import io

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Charger les données depuis un fichier Excel
print("Chargement des données...")
df = pd.read_excel("Data_Train.xlsx")

# Pré-traitement des données
print("Pré-traitement des données...")

# On s'assure que 'Date_of_Journey' est bien au format datetime
# Si des valeurs sont manquantes ou invalides, on met une date par défaut
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce').fillna(pd.Timestamp("1970-01-01"))

# On extrait les heures et minutes depuis 'Duration' et 'Dep_Time'
# Pratique pour analyser les tendances selon l'heure
df['Hours'] = pd.to_datetime(df['Dep_Time'], format='%H:%M', errors='coerce').dt.hour
df['Minutes'] = df['Duration'].str.extract(r'(\d+)m', expand=False).fillna(0).astype(int)

# Découpage de la date en mois et année pour des analyses plus simples
df['Month'] = df['Date_of_Journey'].dt.month
df['Year'] = df['Date_of_Journey'].dt.year

# On supprime les lignes où le prix est manquant
# Car ces lignes ne servent pas à l'entraînement du modèle
df = df.dropna(subset=["Price"])

# Calcul des dépenses totales par mois (utile pour des visualisations)
spending_by_month = df.groupby("Month")["Price"].sum().reset_index()

# Analyse des destinations les plus populaires
# Compte combien de fois chaque destination est mentionnée
popular_destinations = df["Destination"].value_counts().reset_index()
popular_destinations.columns = ["Destination", "Count"]

# Analyse des heures où il y a le plus de départs
peak_hours = df['Hours'].value_counts().reset_index()
peak_hours.columns = ["Hour", "Count"]

# Encodage des colonnes catégoriques (comme Airline, Source, Destination)
# Cela transforme ces colonnes en chiffres pour que le modèle puisse les comprendre
df = pd.get_dummies(df, columns=["Airline", "Source", "Destination"], drop_first=True)

# Préparation des données pour la prédiction
# Sélection des colonnes qui vont servir comme caractéristiques pour le modèle
feature_columns = ["Hours", "Minutes", "Month", "Year"] + [col for col in df.columns if col.startswith(("Airline_", "Source_", "Destination_"))]
X = df[feature_columns]
y = df["Price"]

# On divise les données en deux parties : entraînement et test
# Cela permet de vérifier si le modèle fonctionne bien
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# On entraîne un modèle de régression linéaire
# Ce modèle essaiera de prédire le prix des vols
lr = LinearRegression()
lr.fit(X_train, y_train)

# Évaluation du modèle
# On compare les prédictions du modèle avec les vrais prix
y_pred = lr.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)  # Calcul de l'erreur moyenne quadratique
r2 = r2_score(y_test, y_pred)  # Score R² pour voir la qualité du modèle

# Création des graphiques pour le tableau de bord
def create_monthly_spending_figure():
    # Graphique pour visualiser les dépenses mensuelles
    fig = plt.figure(figsize=(8, 6))
    plt.bar(spending_by_month["Month"], spending_by_month["Price"], color='skyblue')
    plt.title("Dépenses mensuelles")
    plt.xlabel("Mois")
    plt.ylabel("Dépenses totales")
    plt.xticks(spending_by_month["Month"])
    return fig

def create_popular_destinations_figure():
    # Graphique pour les destinations les plus populaires
    fig = plt.figure(figsize=(10, 6))
    plt.bar(popular_destinations["Destination"][:10], popular_destinations["Count"][:10], color='orange')
    plt.title("Top 10 des destinations populaires")
    plt.xlabel("Destination")
    plt.ylabel("Nombre")
    plt.xticks(rotation=45)
    return fig

def create_peak_hours_figure():
    # Graphique pour analyser les heures de pointe
    fig = plt.figure(figsize=(8, 6))
    plt.bar(peak_hours["Hour"], peak_hours["Count"], color='green')
    plt.title("Heures de pointe")
    plt.xlabel("Heure")
    plt.ylabel("Nombre")
    plt.xticks(peak_hours["Hour"])
    return fig

# Fonction pour convertir les graphiques matplotlib en images (base64)
# Cela permet de les afficher dans le tableau de bord
def fig_to_base64(fig):
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    img_str = base64.b64encode(img_bytes.read()).decode()
    return f"data:image/png;base64,{img_str}"

# Mise en page de l'application Dash
app.layout = html.Div([
    html.H1("Tableau de bord : Prédiction des prix des vols"),
    
    html.Div([
        html.H3(f"RMSE : {rmse:.2f}"),  # Affichage de l'erreur moyenne quadratique
        html.H3(f"Score R² : {r2:.2f}")  # Affichage du score R²
    ]),
    
    # Différentes sections pour afficher les graphiques interactifs et statiques
    html.Div([
        html.Div([
            html.H3("Dépenses mensuelles (Interactif)"),
            dcc.Graph(id='monthly-spending-graph'),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Dépenses mensuelles (Image statique)"),
            html.Img(id='monthly-spending-img'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),
    
    # Section pour les destinations populaires
    html.Div([
        html.Div([
            html.H3("Top 10 des destinations populaires (Interactif)"),
            dcc.Graph(id='popular-destinations-graph'),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Top 10 des destinations populaires (Image statique)"),
            html.Img(id='popular-destinations-img'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),
    
    # Section pour les heures de pointe
    html.Div([
        html.Div([
            html.H3("Heures de pointe (Interactif)"),
            dcc.Graph(id='peak-hours-graph'),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Heures de pointe (Image statique)"),
            html.Img(id='peak-hours-img'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
])

# Callback pour mettre à jour les graphiques lorsque l'application est chargée
@app.callback(
    Output('monthly-spending-graph', 'figure'),
    Output('popular-destinations-graph', 'figure'),
    Output('peak-hours-graph', 'figure'),
    Output('monthly-spending-img', 'src'),
    Output('popular-destinations-img', 'src'),
    Output('peak-hours-img', 'src'),
    Input('monthly-spending-graph', 'id')  # L'entrée déclenche l'actualisation
)
def update_graphs(_):
    # On génère les graphiques interactifs et les images statiques
    monthly_spending_figure = create_monthly_spending_figure()
    popular_destinations_figure = create_popular_destinations_figure()
    peak_hours_figure = create_peak_hours_figure()

    # Conversion des graphiques matplotlib en base64
    monthly_spending_img_base64 = fig_to_base64(monthly_spending_figure)
    popular_destinations_img_base64 = fig_to_base64(popular_destinations_figure)
    peak_hours_img_base64 = fig_to_base64(peak_hours_figure)

    # Retourne les graphiques et les images à afficher
    return (
        {
            'data': [{
                'x': spending_by_month["Month"],
                'y': spending_by_month["Price"],
                'type': 'bar',
                'name': 'Dépenses mensuelles',
            }],
            'layout': {
                'title': 'Dépenses mensuelles'
            }
        },
        {
            'data': [{
                'x': popular_destinations["Destination"][:10],
                'y': popular_destinations["Count"][:10],
                'type': 'bar',
                'name': 'Destinations populaires',
            }],
            'layout': {
                'title': 'Top 10 des destinations populaires'
            }
        },
        {
            'data': [{
                'x': peak_hours["Hour"],
                'y': peak_hours["Count"],
                'type': 'bar',
                'name': 'Heures de pointe',
            }],
            'layout': {
                'title': 'Heures de pointe'
            }
        },
        monthly_spending_img_base64,
        popular_destinations_img_base64,
        peak_hours_img_base64
    )

# Lancer l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True)
