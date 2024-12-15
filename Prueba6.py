import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Generar datos simulados más separables
X, y = make_classification(
    n_samples=1000,       # Número de muestras
    n_features=5,         # Número de características (simula sensores EMG)
    n_informative=3,      # Características relevantes para clasificar
    n_redundant=0,        # Características redundantes
    n_clusters_per_class=2,  # Clústeres por clase
    weights=[0.5, 0.5],   # Clases balanceadas
    flip_y=0.1,           # Porcentaje de etiquetas ruidosas
    random_state=42       # Reproducibilidad
)

# 2. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Entrenar el modelo
model = RandomForestClassifier(
    n_estimators=200,    # Más árboles para mejorar el rendimiento
    max_depth=10,        # Limitar la profundidad para evitar sobreajuste
    random_state=42
)
model.fit(X_train, y_train)

# 4. Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# 5. Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Report:")
print(classification_report(y_test, y_pred))

# 6. Generar nuevos datos para predicción
new_data, _ = make_classification(
    n_samples=20,         # Cambiado a 20 muestras
    n_features=5,         # Deben coincidir con las características del modelo
    n_informative=3,
    n_redundant=0,
    random_state=24
)
new_data_scaled = scaler.transform(new_data)  # Escalar igual que los datos de entrenamiento

# Obtener predicciones para los nuevos datos
new_predictions = model.predict(new_data_scaled)

# Mostrar los resultados en consola
results = pd.DataFrame(new_data, columns=[f"Feature_{i+1}" for i in range(5)])
results['Prediction'] = new_predictions
print("\nNuevos datos y predicciones:")
print(results)

# 7. Generar gráficos individuales de señales EMG simuladas
def plot_individual_emg_signals(data, predictions):
    time = np.linspace(0, 1, 100)  # Simula un segundo con 100 puntos
    for i, row in data.iterrows():
        emg_signal = (
            row["Feature_1"] * np.sin(2 * np.pi * 5 * time) +
            row["Feature_2"] * np.cos(2 * np.pi * 10 * time) +
            row["Feature_3"] * np.sin(2 * np.pi * 20 * time) +
            np.random.normal(0, 0.2, len(time))  # Ruido aleatorio
        )
        
        plt.figure(figsize=(8, 4))
        plt.plot(time, emg_signal, label=f"Sample {i+1} (Prediction: {predictions[i]})")
        plt.title(f"Simulated EMG Signal for Sample {i+1}", fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude (normalized)", fontsize=12)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

# Llamar a la función de graficación
plot_individual_emg_signals(results, new_predictions)

# 8. Guardar los resultados en un archivo CSV
results.to_csv("simulated_emg_predictions.csv", index=False)
print("\nPredicciones guardadas en 'simulated_emg_predictions.csv'.")

# Guardar el modelo entrenado
joblib.dump(model, "simulated_emg_model_v2.pkl")
