# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(col={"default payment next month": "default"})
    if "ID" in df.col:
        df = df.drop(col=["ID"])
    # Limpieza necesaria para este dataset
    df = df.loc[df["MARRIAGE"] != 0] 
    df = df.loc[df["EDUCATION"] != 0] 
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    return df

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, compression="zip")

def calculate_confusion_metrics(dataset_name: str, yes, pred) -> dict:
    cm = confusion_matrix(yes, pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


def create_estimator(pipeline: Pipe) -> GridSearchCV:

    param_grid = {
        "classifier__n_estimators": [150],
        "classifier__max_depth": [None],
        "classifier__min_samples_leaf": [1, 2]
    }

def calculate_precision_metrics(dataset_name: str, yes, pred) -> dict:
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": float(precision_score(yes, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yes, pred)),
        "recall": float(recall_score(yes, pred, zero_division=0)),
        "f1_score": float(f1_score(yes, pred, zero_division=0)),
    }

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=5, 
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

def save_model(path: str, estimar: GridSearchCV):
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    with gzip.open(path, "wb") as f:
        pickle.dump(estimar, f)

def create_pipeline() -> Pipeline:
    features = ["SEX", "EDUCATION", "MARRIAGE"]
    processor = Columnformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), features)],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessor", processor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

def main():
    input_path_file = "files/input/"
    models_path_file = "files/models/"
    output_path_file = "files/output/"

    test_df = load_dataset(os.path.join(input_path_fil, "test_data.csv.zip"))
    train_df = load_dataset(os.path.join(input_path_fil, "train_data.csv.zip"))

    test_df = clean_dataset(test_df)
    train_df = clean_dataset(train_df)

    test_x = test_df.drop(columns=["default"])
    test_y = test_df["default"]

    train_x = train_df.drop(columns=["default"])
    train_y = train_df["default"]

    pipeline = create_pipeline()
    estimator = create_estimator(pipeline)
    estimator.fit(x_train, y_train)

    save_model(os.path.join(models_path_file, "model.pkl.gz"), estimator)

    test_pred_y = estimator.predict(test_x)
    precision_test_metrics = calculate_precision_metrics("test", test_y, test_pred_y)
    train_pred_y = estimator.predict(train_x)
    precision_train_metrics = calculate_precision_metrics("train", train_y, train_pred_y)

    test_confusion_metrics = calculate_confusion_metrics("test", test_y, test_pred_y)
    train_confusion_metrics = calculate_confusion_metrics("train", train_y, train_pred_y)

    os.makedirs(output_path_file, exist_ok=True)
    with open(os.path.join(output_files_path, "metrics.json"), "w") as file:
        file.write(json.dumps(precision_train_metrics) + "\n")
        file.write(json.dumps(precision_test_metrics) + "\n")
        file.write(json.dumps(train_confusion_metrics) + "\n")
        file.write(json.dumps(test_confusion_metrics) + "\n")

if __name__ == "__main__":
    main()