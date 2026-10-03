# [PYSPARK ML / MASTER EN BIG DATA]

Este repositorio contiene la resolución de la práctica guiada de PYSPARK ML y modelado predictivo.

## 📝 Enunciado de la Práctica

Dado el siguiente dataset:

https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv

Cuyas variables representan:

age: edad del beneficiario principal
sex: género del contratante del seguro, femenino, masculino
bmi: índice de masa corporal, que proporciona una comprensión del cuerpo, de los pesos que son relativamente altos o bajos en relación con la altura, índice objetivo de peso corporal (kg / m ^ 2) utilizando la relación entre la altura y el peso, idealmente entre 18,5 y 24,9.
children: número de hijos cubiertos por el seguro de salud / número de dependientes
smoker: si fuma o no
region: área de residencia del beneficiario en EE. UU., noreste, sureste, suroeste, noroeste.
charges: costos médicos individuales facturados por el seguro de salud.


Utilizando la API de pandas de PySpark

Realiza un EDA del dataset anterior.

Luego conviertelós a un DataFrame de Spark normal, realiza las conversiones necesarias para poder hacer machine learning (StringIndexer, StandardScaler, VectorAssembler...) y aplica un modelo de regresión lineal usando pyspark.ml

La variable objetivo a predecir sería charges.

## 🗂️ Estructura del Repositorio

*   **`insurance.csv`**: Conjunto de datos (dataset) utilizado para el análisis.
*   **`Practica_final_pyspark_Fer_mez_FINAL`**: Jupyter Notebook que contiene todo el Análisis Exploratorio de Datos (EDA), visualizaciones y conclusiones.

## 🚀 Requisitos y Configuración
Para replicar este proyecto localmente, asegúrate de tener instalado Python y las siguientes librerías:
*   Pyspark.pandas
*   Matplotlib / Seaborn
*   Scikit-learn
*   Pyspark.pandas.ml
*   Os


