import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Criando uma classe de pré-processamento
class pre_preproc:

    def __init__(self, df, seed=None):
        self.df = df
        self.seed = seed
        self.pipe = None  # Atributo para armazenar o pipeline

    def train_test_split(self, feature_columns, target_column, test_size=0.20):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        X = self.df[feature_columns]
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.seed,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test

    def pipe_process(self, X_train, X_test):
        # Definindo o pipeline e armazenando como atributo da instância
        self.pipe = Pipeline(steps=[
            ('encoder', OneHotEncoder(sparse_output=False)),
            ('scaler', StandardScaler())
        ])

        # Ajuste e transformação dos dados
        X_train_processed = self.pipe.fit_transform(X_train)
        X_test_processed = self.pipe.transform(X_test)

        return X_train_processed, X_test_processed

    def apply_smote(self, X_train, y_train):
        smote = SMOTE(random_state=self.seed)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled
