import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, month, year, to_date, regexp_extract, substring, dayofweek
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import base64
import io

# Initialisation de la session Spark
spark = SparkSession.builder \
    .appName("Prediction des prix des vols") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Charger les donnees depuis un fichier Excel
print("Chargement des donnees...")
df = pd.read_excel("Data_Train.xlsx")  # Charger les donnees Ã  l'aide de pandas

# Convertir le DataFrame pandas en DataFrame PySpark
spark_df = spark.createDataFrame(df)

# Pretraitement des donnees
print("Pretraitement des donnees...")

# Conversion de 'Date_of_Journey' en format date
spark_df = spark_df.withColumn("Date_of_Journey", to_date("Date_of_Journey", "dd/MM/yyyy"))

# Extraction du mois, de l'annee et du jour de la semaine depuis 'Date_of_Journey'
spark_df = spark_df.withColumn("Month", month(spark_df["Date_of_Journey"]))
spark_df = spark_df.withColumn("Year", year(spark_df["Date_of_Journey"]))
spark_df = spark_df.withColumn("Weekday", dayofweek(spark_df["Date_of_Journey"]))  # Dimanche = 1, Samedi = 7

# Creation d'une colonne 'Weekend' (1 si samedi/dimanche, sinon 0)
spark_df = spark_df.withColumn("Weekend", when((col("Weekday") == 7) | (col("Weekday") == 1), 1).otherwise(0))

# Extraction des heures depuis 'Dep_Time'
spark_df = spark_df.withColumn("Hours", substring("Dep_Time", 1, 2).cast("int"))

# Extraction des minutes depuis 'Duration' (en supposant le format '2h 50m')
spark_df = spark_df.withColumn("Minutes", regexp_extract("Duration", r'(\d+)m', 1).cast("int"))

# Conversion de 'Total_Stops' en valeurs numeriques
# Ex : "non-stop" = 0, "1 stop" = 1, etc.
spark_df = spark_df.withColumn("Stops", when(col("Total_Stops") == "non-stop", 0)
                               .when(col("Total_Stops").like("%1%"), 1)
                               .when(col("Total_Stops").like("%2%"), 2)
                               .when(col("Total_Stops").like("%3%"), 3)
                               .when(col("Total_Stops").like("%4%"), 4)
                               .otherwise(0))

# Gestion des valeurs manquantes
spark_df = spark_df.fillna({'Hours': 0, 'Minutes': 0, 'Stops': 0}).dropna(subset=["Price"])

# Encodage des colonnes categoriques (Airline, Source, Destination)
categorical_columns = ["Airline", "Source", "Destination"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index") for col in categorical_columns]
encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_Vec") for col in categorical_columns]

# Assemblage des caracteristiques en un seul vecteur
assembler = VectorAssembler(
    inputCols=["Hours", "Minutes", "Stops", "Month", "Year", "Weekday", "Weekend"] + [col + "_Vec" for col in categorical_columns],
    outputCol="features"
)

# Normalisation des caracteristiques
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Definition du modÃ¨le de regression lineaire
lr = LinearRegression(featuresCol="scaledFeatures", labelCol="Price", regParam=0.01, elasticNetParam=0.8, maxIter=100)

# Creation du pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, lr])

# Division des donnees en ensemble d'entraÃ®nement et de test
train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)

# EntraÃ®nement du modÃ¨le
print("EntraÃ®nement du modÃ¨le...")
pipeline_model = pipeline.fit(train_data)

# Ã‰valuation du modÃ¨le
print("Ã‰valuation du modÃ¨le...")
predictions = pipeline_model.transform(test_data)

# Calcul des metriques d'evaluation
evaluator_rmse = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

print(f"Erreur quadratique moyenne (RMSE) : {rmse}")
print(f"Score RÂ² : {r2}")

# Extraction des donnees pour les visualisations
spending_by_month = spark_df.groupBy("Month").agg({"Price": "sum"}).toPandas()
popular_destinations = spark_df.groupBy("Destination").count().toPandas().sort_values("count", ascending=False)
peak_hours = spark_df.groupBy("Hours").count().toPandas().sort_values("count", ascending=False)

# Fonctions de visualisation
def create_monthly_spending_figure():
    fig = plt.figure(figsize=(8, 6))
    plt.bar(spending_by_month["Month"], spending_by_month["sum(Price)"], color='skyblue')
    plt.title("Depenses mensuelles")
    plt.xlabel("Mois")
    plt.ylabel("Depenses totales")
    plt.xticks(spending_by_month["Month"])
    return fig

def create_popular_destinations_figure():
    fig = plt.figure(figsize=(10, 6))
    plt.bar(popular_destinations["Destination"][:10], popular_destinations["count"][:10], color='orange')
    plt.title("Top 10 des destinations populaires")
    plt.xlabel("Destination")
    plt.ylabel("Nombre")
    plt.xticks(rotation=45)
    return fig

def create_peak_hours_figure():
    fig = plt.figure(figsize=(8, 6))
    plt.bar(peak_hours["Hours"], peak_hours["count"], color='green')
    plt.title("Heures de pointe")
    plt.xlabel("Heure")
    plt.ylabel("Nombre")
    plt.xticks(peak_hours["Hours"])
    return fig

def fig_to_base64(fig):
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    img_str = base64.b64encode(img_bytes.read()).decode()
    return f"data:image/png;base64,{img_str}"

# Mise en page de l'application Dash
app.layout = html.Div([
    html.H1("Tableau de bord : Prediction des prix des vols"),
    
    html.Div([
        html.H3(f"RMSE : {rmse:.2f}"),
        html.H3(f"Score RÂ² : {r2:.2f}")
    ]),
    
    # Section des depenses mensuelles
    html.Div([
        html.Div([
            html.H3("Depenses mensuelles (Interactif)"),
            dcc.Graph(id='monthly-spending-graph'),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Depenses mensuelles (Image statique)"),
            html.Img(id='monthly-spending-img'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),
    
    # Section des destinations populaires
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
    
    # Section des heures de pointe
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

@app.callback(
    Output('monthly-spending-graph', 'figure'),
    Output('popular-destinations-graph', 'figure'),
    Output('peak-hours-graph', 'figure'),
    Output('monthly-spending-img', 'src'),
    Output('popular-destinations-img', 'src'),
    Output('peak-hours-img', 'src'),
    Input('monthly-spending-graph', 'id')
)
def update_graphs(_):
    # Generation des graphiques pour chaque section
    monthly_spending_figure = create_monthly_spending_figure()
    popular_destinations_figure = create_popular_destinations_figure()
    peak_hours_figure = create_peak_hours_figure()

    # Conversion des graphiques matplotlib en images (base64)
    monthly_spending_img_base64 = fig_to_base64(monthly_spending_figure)
    popular_destinations_img_base64 = fig_to_base64(popular_destinations_figure)
    peak_hours_img_base64 = fig_to_base64(peak_hours_figure)

    # Retourne les graphiques et les images pour l'affichage dans Dash
    return (
        {
            'data': [{
                'x': spending_by_month["Month"],
                'y': spending_by_month["sum(Price)"],
                'type': 'bar',
                'name': 'Depenses mensuelles',
            }],
            'layout': {
                'title': 'Depenses mensuelles'
            }
        },
        {
            'data': [{
                'x': popular_destinations["Destination"][:10],
                'y': popular_destinations["count"][:10],
                'type': 'bar',
                'name': 'Destinations populaires',
            }],
            'layout': {
                'title': 'Top 10 des destinations populaires'
            }
        },
        {
            'data': [{
                'x': peak_hours["Hours"],
                'y': peak_hours["count"],
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

# Lancement de l'application Dash
if __name__ == '__main__':
    app.run_server(debug=True)
