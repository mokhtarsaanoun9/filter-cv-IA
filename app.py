import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def create_time_series_model(dataset):
    # Charger le jeu de données
    data = pd.read_csv(dataset)  # Vous pouvez ajuster cette ligne selon le format de votre jeu de données

    # Prétraitement des données
    # Suppose que votre jeu de données a une colonne 'date' et une colonne 'valeur'
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # Préparer les données pour la série temporelle (exemple avec la lateness)
    data['Lateness'] = (data['StartTime'] - pd.Timestamp('09:00:00')).dt.total_seconds() / 60
    daily_lateness = data.resample('D').mean()['Lateness']
    daily_lateness.dropna(inplace=True)

    # Décomposer la série temporelle pour observer la tendance, la saisonnalité et les résidus
    decomposition = sm.tsa.seasonal_decompose(daily_lateness, model='additive')
    fig = decomposition.plot()
    plt.show()

    # Créer un modèle SARIMA (Seasonal Autoregressive Integrated Moving Average)
    mod = sm.tsa.statespace.SARIMAX(daily_lateness,
                                     order=(1, 0, 1),
                                     seasonal_order=(1, 1, 1, 7),
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)

    results = mod.fit()

    # Résumé du modèle
    print(results.summary())

    # Plot des diagnostics
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()

    # Créer également un modèle de régression RandomForest pour la comparaison
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train_data.index.values.reshape(-1, 1), train_data['valeur'])
    rf_predictions = rf_model.predict(test_data.index.values.reshape(-1, 1))
    rf_mse = mean_squared_error(test_data['valeur'], rf_predictions)
    print(f"Erreur quadratique moyenne pour RandomForest : {rf_mse}")

# Exemple d'utilisation
dataset_path = 'chemin/vers/votre/dataset.csv'
create_time_series_model(dataset_path)
