# Práctica final PySpark: Insurance

A partir del [dataset de costes médicos de un seguro de salud](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv), se resuelven dos problemas con PySpark:

- **Análisis exploratorio (EDA):** usando la API de pandas de PySpark (`pyspark.pandas`).
- **Regresión:** convertir los datos a un DataFrame de Spark, prepararlos para machine learning (StringIndexer, OneHotEncoder, VectorAssembler...) y predecir los costes médicos (`charges`) con `pyspark.ml`.

## Estructura del proyecto

```
├── insurance.csv                                Datos en crudo de los asegurados
└── Practica_final_pyspark_Fer_mez_FINAL.ipynb   Análisis exploratorio y modelos de regresión con pyspark.ml
```

## Requisitos

Python 3 con pyspark, numpy 1.26.4, matplotlib y seaborn, y Java 8 (el notebook está preparado para Google Colab e instala pyspark, numpy y openjdk-8 en la primera celda).

## Ejecución

Abrir el notebook en Google Colab, subir `insurance.csv` y ejecutar las celdas en orden.

## Datos

| Variable | Descripción |
|---|---|
| `age` | Edad del beneficiario principal |
| `sex` | Género del contratante del seguro (femenino, masculino) |
| `bmi` | Índice de masa corporal (kg / m²), idealmente entre 18,5 y 24,9 |
| `children` | Número de hijos cubiertos por el seguro de salud / número de dependientes |
| `smoker` | Si fuma o no |
| `region` | Área de residencia del beneficiario en EE. UU. (noreste, sureste, suroeste, noroeste) |
| `charges` | Costes médicos individuales facturados por el seguro de salud (variable objetivo) |

## Modelos

| Problema | Modelo | Transformaciones | Resultado en test |
|---|---|---|---|
| Costes médicos | Regresión Lineal con CrossValidator | StringIndexer y OneHotEncoder de smoker; VectorAssembler de age, bmi y smoker | MAE ≈ 4.330 (≈ 30 % de error sobre una media de unos 13.000) |
| Costes médicos | Random Forest con CrossValidator | Las mismas que en la Regresión Lineal | MAE casi 300 unidades más bajo que la Regresión Lineal |

Se elige el **Random Forest**, ya que de media se equivoca menos.

## Notebook

En [Practica_final_pyspark_Fer_mez_FINAL.ipynb](Practica_final_pyspark_Fer_mez_FINAL.ipynb) se hace el análisis exploratorio y se entrenan los modelos:

- **Carga y descripción de los datos:** el dataset está completo y no hay valores nulos.
- **Análisis univariante:** distribución de edad, BMI, hijos, sexo, fumador, región y costes.
- **Análisis bivariante:** relación de cada variable con los costes. El **BMI**, la **edad** y el hecho de ser **fumador** son determinantes, mientras que el sexo, la región y el número de hijos no influyen de forma relevante.
- **Regresión lineal:** conversión a DataFrame de Spark, StringIndexer, OneHotEncoder y VectorAssembler, y búsqueda de hiperparámetros con ParamGridBuilder y CrossValidator.
- **Random Forest:** mismo proceso con RandomForestRegressor y comparación con la Regresión Lineal.
