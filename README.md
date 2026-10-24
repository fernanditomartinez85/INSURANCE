# 🏥 Insurance: costes de un seguro médico

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square) ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square) ![Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

> Práctica final de PySpark: análisis exploratorio y predicción de los costes médicos de cada asegurado.

## 🎯 Objetivo

A partir del [dataset de costes médicos de un seguro de salud](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv), se resuelven dos problemas con PySpark:

- **Análisis exploratorio (EDA):** usando la API de pandas de PySpark (`pyspark.pandas`).
- **Regresión:** convertir los datos a un DataFrame de Spark, prepararlos para machine learning (StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler...) y predecir los costes médicos (`charges`) con `pyspark.ml`.

## 📊 Datos

Dataset público de costes médicos de un seguro de salud en EE. UU. ([fuente](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)). Está completo y no tiene valores nulos.

| Variable | Descripción |
|---|---|
| `age` | Edad del beneficiario principal |
| `sex` | Género del contratante del seguro (femenino, masculino) |
| `bmi` | Índice de masa corporal (kg / m²), idealmente entre 18,5 y 24,9 |
| `children` | Número de hijos cubiertos por el seguro de salud / número de dependientes |
| `smoker` | Si fuma o no |
| `region` | Área de residencia del beneficiario en EE. UU. (noreste, sureste, suroeste, noroeste) |
| `charges` | Costes médicos individuales facturados por el seguro de salud (variable objetivo) |

## 🗂️ Estructura del proyecto

```
├── insurance.csv                                Datos en crudo de los asegurados
└── Practica_final_pyspark_Fer_mez_FINAL.ipynb   Análisis exploratorio y modelos de regresión con pyspark.ml
```

## ⚙️ Requisitos

Python 3 con **pyspark 4.0.4**, matplotlib y seaborn, y **Java 17**. La versión 4.0.0 de pyspark falla en Windows, por eso se fija la 4.0.4.

En Google Colab no hace falta instalar nada: la primera celda detecta que está en Colab e instala pyspark 4.0.4 y Java 17.

## ▶️ Ejecución

- **Google Colab:** abrir el notebook, subir `insurance.csv` y ejecutar las celdas en orden.
- **En local:** instalar pyspark 4.0.4 (`pip install pyspark==4.0.4`) y Java 17, y ejecutar el notebook desde la carpeta del proyecto.

## 🤖 Modelo

| Problema | Modelo | Transformaciones | Resultado en test |
|---|---|---|---|
| Costes médicos | Regresión Lineal con CrossValidator (5 folds) | StringIndexer y OneHotEncoder de smoker; VectorAssembler de age, bmi y smoker; StandardScaler en un Pipeline junto a la regresión | MAE ≈ 4.288 dólares (≈ 32 % del cargo medio). Mejores hiperparámetros: regParam 2.0, elasticNetParam 1.0, maxIter 10 |
| Costes médicos | Random Forest con CrossValidator (5 folds) | Las mismas, sin escalado (los árboles no lo necesitan) | **MAE ≈ 3.353 dólares** (≈ 25 % del cargo medio). Mejores hiperparámetros: numTrees 50, maxDepth 20 |

Los dos modelos se entrenan y evalúan con **el mismo reparto** de entrenamiento (80 %) y prueba (20 %). Se elige el **Random Forest**: se equivoca de media 934 dólares menos (un 22 % menos de error).

## 📓 Notebook

En [Practica_final_pyspark_Fer_mez_FINAL.ipynb](Practica_final_pyspark_Fer_mez_FINAL.ipynb) se hace el análisis exploratorio y se entrenan los modelos:

- **Carga y descripción de los datos:** no hay valores nulos y se elimina un registro duplicado. Matriz de correlación incluyendo fumador (0/1).
- **Análisis univariante:** distribución de edad, BMI, hijos, sexo, fumador, región y costes.
- **Análisis bivariante:** relación de cada variable con los costes, separando fumadores y no fumadores, y porcentaje de fumadores por sexo, hijos y región.
- **Regresión lineal:** conversión a DataFrame de Spark, StringIndexer, OneHotEncoder, VectorAssembler y StandardScaler; búsqueda de hiperparámetros con ParamGridBuilder y CrossValidator; mejores hiperparámetros y coeficientes.
- **Random Forest:** mismo proceso con RandomForestRegressor y los mismos datos de entrenamiento y prueba, y comparación con la Regresión Lineal.
- **Conclusiones:** hallazgos, limitaciones y próximos pasos.

## 💡 Conclusiones

- Lo que más encarece el seguro es **ser fumador**: los fumadores pagan casi 4 veces más (32.050 frente a 8.441 de media).
- La **edad** sube los cargos en todos los clientes, y el **BMI** solo encarece de verdad en los fumadores: un fumador con BMI por encima de 30 paga entre 35.000 y 50.000.
- El sexo, la región y el número de hijos apenas influyen: sus diferencias se explican por la proporción de fumadores de cada grupo.
- El **Random Forest** se equivoca menos que la Regresión Lineal (MAE de 3.353 frente a 4.288 dólares), así que es el modelo elegido.
