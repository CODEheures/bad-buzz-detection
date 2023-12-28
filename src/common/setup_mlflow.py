import mlflow
import streamlit as st
from common import params
from streamlit import session_state as ss
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository


def init_mlflow():
    """Initialized Mlflow
    """
    mlflow.set_tracking_uri(params.tracking_uri)
    mlflow.tracking.MlflowClient(tracking_uri=params.tracking_uri)


def init_tracking(experiment_name: str):
    """Initialized the tracking on train

    Args:
        experiment_name (str): Name of the experiment; Ex: air-paradis
    """
    init_mlflow()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            s3_bucket = params.s3_uri
            experiment = mlflow.create_experiment(experiment_name, s3_bucket)
        except Exception:
            st.error(f"""MlFlow: Impossible de récupérer ou créer l\'experience MlFlow.
                     Vérifier l\'état du serveur à l\'adresse suivante: {params.tracking_uri}""")

    if (experiment is not None):
        mlflow.set_experiment(experiment_name=experiment_name)
        ss['mlflow_ready'] = True
        xp_path = f'#/experiments/{experiment.experiment_id}'
        st.success(f"Mlflow initialisé avec succès! Visitez l\'adresse suivante: {params.tracking_uri}{xp_path}")


def publish_model():
    """Create a new model version of the current run on MlFlow register
    """
    run: mlflow.ActiveRun = ss['run']
    model = ss['selected_model']
    user_name = ss['user_name']
    description = ss['description']

    client = mlflow.client.MlflowClient(tracking_uri=params.tracking_uri)

    model_src = RunsArtifactRepository.get_underlying_uri(f"runs:/{run.info.run_id}/model")
    st.write(model_src)
    client.create_model_version(name=params.model_name,
                                source=model_src,
                                run_id=run.info.run_id,
                                description=description,
                                tags={'model': model,
                                      'entraineur': user_name})

    st.success(f"""Model enregistré dans le registre des models.
                Promouvoir celui en production en suivant ce lien {params.tracking_uri}""")
