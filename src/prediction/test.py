import mlflow.pyfunc
from common import params, setup_mlflow
import streamlit as st
import pandas as pd
import boto3
from sklearn.metrics import accuracy_score
from mlflow.entities.model_registry import ModelVersion

setup_mlflow.init_mlflow()
client = mlflow.MlflowClient()
models = client.search_model_versions("name='air-paradis'")


def format_select_model(model: ModelVersion) -> str:
    """Format model to string to display on select option

    Args:
        model (ModelVersion): The model to format

    Returns:
        str: The formatting model to string
    """
    label = '*' if 'production' in model.aliases else ''
    label += f"Version {model.version}"
    label += f": {model.description}"
    return label


def run():
    """Entry point to display Test page
    """
    st.header("Test model", divider='rainbow')

    selected_model = st.selectbox('Evaluer un model avec le jeu de test',
                                  [model for model in models],
                                  format_func=format_select_model,
                                  placeholder='Choisissez un model à évaluer')

    if selected_model is not None:
        try:
            run = mlflow.get_run(selected_model.run_id)
            df_test_uri = run.info.artifact_uri + f'/df_test/{run.info.run_id}.csv'
            file_name = df_test_uri.split(params.s3_bucket + '/')[1]
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=params.s3_bucket, Key=file_name)
            df = pd.read_csv(obj['Body'])
            df = df[['text', 'target']]
            X = df.drop(columns=['target'])  # noqa: N806
            y = df['target']

            version = mlflow.pyfunc.load_model(f"models:/{selected_model.name}/{selected_model.version}")
            predict = version.predict(X)

            score = accuracy_score(y.to_list(), list(predict))
            st.success(f'Score du model selectionné sur jeu de test: {score:.3f}')

            st.divider()
            st.write('Jeu de test:')
            st.write(df)
        except Exception:
            st.error('Jeu de test non disponible pour ce model')
