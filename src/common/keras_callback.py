import math
import streamlit as st
from streamlit import session_state as ss
from keras.callbacks import Callback


class TrainCallback(Callback):
    def __init__(self):
        self.current_epoch = None
        self.batch_count = math.ceil(len(ss['X_train'])/ss['batch_size'])

    def on_train_begin(self, logs=None):
        st.write("Début de l'entrainement")

    def on_train_end(self, logs=None):
        st.write("Fin de l'entrainement")

    def on_epoch_begin(self, epoch, logs=None):
        st.write(f"Epoch {epoch+1}/{ss['epochs']}")
        self.current_epoch = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        with self.current_epoch.container():
            text = ''
            for key in list(logs.keys()):
                text = text + f" - {key}={logs[key]:.5f}"
            st.progress(value=1.0, text=f"batch {self.batch_count}/{self.batch_count} - {text}")

    def on_train_batch_begin(self, batch, logs=None):
        with self.current_epoch.container():
            st.progress(value=batch/self.batch_count, text=f"batch {batch+1}/{self.batch_count}")

    def on_train_batch_end(self, batch, logs=None):
        with self.current_epoch.container():
            text = ''
            for key in list(logs.keys()):
                text = text + f" - {key}={logs[key]:.5f}"
            st.progress(value=batch/self.batch_count, text=f"batch {batch+1}/{self.batch_count} - {text}")
