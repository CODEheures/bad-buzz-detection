import mlflow
import streamlit as st


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
            st.error('Impossible de créer l\'experience MlFlow.')
            st.stop()

    mlflow.set_experiment(experiment_name=experiment_name)
