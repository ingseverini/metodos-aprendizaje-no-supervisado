import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

#Lee el dataset
df = pd.read_csv("rutas-transporte.csv", sep=";", encoding="latin-1")

#Limpia nombres de columnas
df.columns = df.columns.str.strip()

#Codificar variables categóricas
le = LabelEncoder()
for col in ['Tipo de vehículo', 'Ubicación geográfica', 'Estado del vehículo']:
    df[col] = le.fit_transform(df[col])

# Asegurar que las columnas numéricas sean tipo numérico
columnas_numericas = ['Frecuencia (min)', 'Cantidad de pasajeros diarios',
                      'Duración del viaje (min)', 'Distancia (km)']
for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con valores nulos
df.dropna(inplace=True)

#Selecciona variables para el modelo
variables = columnas_numericas + ['Tipo de vehículo', 'Ubicación geográfica']
X = df[variables]

#Escala los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Aplica KMeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df['Grupo'] = kmeans.fit_predict(X_scaled)

#Imprime los resultados antes de la visualizarse
print("Iniciando la visualización del gráfico...\n")
print("\nCantidad de rutas por grupo:")
print(df['Grupo'].value_counts())

#Centroide del modelo (en datos escalados)
print("\nCentroides de los grupos (valores normalizados):")
print(kmeans.cluster_centers_)

#Mustra las primeras filas con el grupo asignado
print("\nPrimeras filas del dataset con grupo asignado:")
print(df[['Ruta', 'Grupo']].head())

#Visualiza resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Distancia (km)', y='Cantidad de pasajeros diarios',
                hue='Grupo', palette='Set2', s=100)
plt.title('Agrupamiento de rutas de transporte')
plt.xlabel('Distancia (km)')
plt.ylabel('Cantidad de pasajeros diarios')
plt.grid(True)
plt.tight_layout()

#Asegura que el gráfico se muestra correctamente en entornos de consola
plt.show()

#Confirma que el archivo ha sido guardado
df.to_csv("rutas_agrupadas.csv", index=False)
print("\nArchivo 'rutas_agrupadas.csv' guardado con éxito.")
