import mlflow
import streamlit as st
from streamlit import session_state as ss


def init_tracking(experiment_name):
    tracking_uri = "https://mlflow.air-paradis.codeheures.fr"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            s3_bucket = "s3://codeheures/mlflow-air-paradis/"
            experiment = mlflow.create_experiment(experiment_name, s3_bucket)
        except Exception:
            st.error(f"""MlFlow: Impossible de récupérer ou créer l\'experience MlFlow.
                     Vérifier l\'état du serveur à l\'adresse suivante: {tracking_uri}""")

    if (experiment is not None):
        mlflow.set_experiment(experiment_name=experiment_name)
        ss['mlflow_ready'] = True
        xp_path = f'#/experiments/{experiment.experiment_id}'
        st.success(f"Mlflow initialisé avec succès! Visitez l\'adresse suivante: {tracking_uri}{xp_path}")
