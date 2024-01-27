import os
import streamlit as st
from st_pages import add_page_title
from streamlit import session_state as ss
import mlflow
import numpy as np
import pandas as pd
from common import params
from streamlit_extras.switch_page_button import switch_page
from experiment import pages_management
from common.keras_callback import TrainCallback
from keras.callbacks import EarlyStopping
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, precision_score, roc_curve, auc
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from transformers import pipeline, Trainer, AutoTokenizer


add_page_title()
st.header("Cette page permet d'entrainer un model et d'obtenir les scores et graphiques associés", divider='rainbow')

with st.status("Preprocess data...", expanded=True) as status:
    if (('train_ok' not in ss) or (ss['train_ok'] is False)):
        mlflow.end_run()
        start_time = time.time()

        model = ss['model']
        X_train = ss['X_train']
        X_validation = ss['X_validation']
        X_test = ss['X_test']
        y_train = ss['y_train']
        y_validation = ss['y_validation']
        y_test = ss['y_test']

        with mlflow.start_run() as run:
            if (ss['selected_model'] == params.model_enum.SVM):
                mlflow.sklearn.autolog()
                st.markdown('1. Entrainement')
                model.fit(X_train, y_train)
                time_delta = timedelta(seconds=round((time.time() - start_time), 0))

                st.write([f"{key}: {value:.4f}"
                         for key, value
                         in mlflow.get_run(run_id=run.info.run_id).data.metrics.items()])

                # TESTS
                y_score = model.decision_function(X_test)
                predictions = model.predict(X_test)

            elif (ss['selected_model'] == params.model_enum.Tensorflow_Keras_base_embedding) \
                    or (ss['selected_model'] == params.model_enum.Tensorflow_Keras_base_LSTM_embedding):
                mlflow.tensorflow.autolog()
                mlflow.log_param('embedding', ss['embedding'])
                st.markdown('1. Entrainement')
                model.fit(X_train,
                          y_train,
                          validation_data=(X_validation, y_validation),
                          epochs=ss['epochs'],
                          batch_size=ss['batch_size'],
                          verbose=0,
                          callbacks=[TrainCallback(),
                                     EarlyStopping(monitor='val_auc_1',
                                                   mode='max',
                                                   patience=3,
                                                   restore_best_weights=True)]
                          )
                time_delta = timedelta(seconds=round((time.time() - start_time), 0))

                st.write([f"{key}: {value:.4f}"
                         for key, value
                         in mlflow.get_run(run_id=run.info.run_id).data.metrics.items()])

                # TESTS
                y_score = model.predict(X_test).reshape(-1)
                predictions = np.where(y_score < 0.5, 0, 1)

            elif (ss['selected_model'] == params.model_enum.BERT_Transfert_learning):
                mlflow.transformers.autolog()
                trainer: Trainer = model
                trainer.train()
                time_delta = timedelta(seconds=round((time.time() - start_time), 0))

                tuned_pipeline = pipeline(task="text-classification",
                                          model=trainer.model,
                                          batch_size=8,
                                          tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
                                          device="cpu",
                                          )
                model_config = {"batch_size": 8}
                signature = mlflow.models.infer_signature(
                    ["This is a tweet!", "And this is also a tweet."],
                    mlflow.transformers.generate_signature_output(
                        tuned_pipeline, ["This is a tweet response!", "So is this."]
                    ),
                    params=model_config,
                )
                model_info = mlflow.transformers.log_model(transformers_model=tuned_pipeline,
                                                           artifact_path="model",
                                                           signature=signature,
                                                           input_example=["This is a good day", "This is a sa day"],
                                                           model_config=model_config,
                                                           )

                # TESTS
                pipeline_score = tuned_pipeline(list(X_test))
                y_score = pd.DataFrame(pipeline_score)
                y_score['inversed'] = y_score.apply(lambda x: x['score'] if x['label'] == 1 else (1 - x['score']), axis=1)
                y_score = list(y_score['inversed'])
                predictions = list(pd.DataFrame(pipeline_score)['label'])

            mlflow.set_tag('model', params.get_format_model_short(ss['selected_model']))
            mlflow.log_metrics({'fit_time': time_delta.seconds})
            ss['time_delta'] = time_delta

            st.markdown('2. Test')
            test_precision = precision_score(y_test, predictions)
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            test_auc = auc(fpr, tpr)
            cm = confusion_matrix(y_test, predictions, normalize='true', labels=[0, 1])
            fig = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            fig.plot(cmap=plt.cm.Blues)
            plt.title('Normalized confusion matrix')
            plt.savefig('test_confusion_matrix.png')

            fig = RocCurveDisplay(fpr=fpr,
                                  tpr=tpr,
                                  roc_auc=test_auc,
                                  estimator_name=params.get_format_model(ss['selected_model']))
            fig.plot()
            plt.title('ROC curve')
            plt.savefig('test_roc_curve.png')

            mlflow.log_metrics({'test_precision': test_precision})
            mlflow.log_metrics({'test_auc': test_auc})
            mlflow.log_artifact('test_confusion_matrix.png')
            mlflow.log_artifact('test_roc_curve.png')

            ss['score'] = test_auc
            st.write(f"AUC score du model: {ss['score']:.4f}")

            df_test = pd.DataFrame(X_test, columns=['text'])
            df_test['target'] = y_test

            file_name = f'{run.info.run_id}.csv'
            df_test.to_csv(file_name, index=False)
            mlflow.log_artifact(file_name, 'df_test')
            os.remove(file_name)
            uri = mlflow.get_artifact_uri('df_test') + '/' + file_name
            dataset = mlflow.data.from_pandas(df_test, uri)
            mlflow.log_input(dataset, context='test')
            ss['run'] = run

            st.write(f"Entrainement fini en {ss['time_delta']}")
            status.update(label="Fin du traitement", state="complete", expanded=True)
            ss['train_ok'] = True

    else:
        st.write([f"{key}: {value:.4f}"
                  for key, value
                  in mlflow.get_run(run_id=ss['run'].info.run_id).data.metrics.items()])
        st.markdown('2. Validation')
        st.write(f"Score du model: {ss['score']:.4f}")
        st.write(f"Entrainement fini en {ss['time_delta']}")
        status.update(label="Fin du traitement", state="complete", expanded=True)


if (ss['train_ok']):
    pages_management.update_pages()
    user_name = st.text_input('Votre nom', placeholder='Renseigner votre nom...')
    description = st.text_input('Description de l\'entrainement', placeholder='Donner une description de cet entrainement...')
    if user_name and description and st.button("Publier ce model"):
        ss['user_name'] = user_name
        ss['description'] = description
        switch_page("Publication model")
