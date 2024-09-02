import pandas as pd
from classes_preprocess import pre_preproc
from classes_eda import SaveDf

# Importando o DF após análise exploratória
df_pos_eda = pd.read_csv('C:/Users/Gdper/health_care/.venv/df_pos_eda.csv')
pd.set_option('display.max_columns', None)

# Definindo as colunas de features e a coluna alvo e Realizando o train_test_split
preprocessor = pre_preproc(df_pos_eda, seed=42)

feature_columns = df_pos_eda.drop(columns=['Test Results']).columns.tolist()
target_column = 'Test Results'

X_train, X_test, y_train, y_test = preprocessor.train_test_split(feature_columns, target_column)

# Aplicar o pipeline (OneHotEncoder + StandardScaler)
X_train_processed, X_test_processed = preprocessor.pipe_process(X_train, X_test)

# Aplicar SMOTE para balancear o conjunto de treino
X_train_resampled, y_train_resampled = preprocessor.apply_smote(X_train_processed, y_train)

# Obtendo os nomes das colunas codificadas
encoded_columns = preprocessor.pipe.named_steps['encoder'].get_feature_names_out(input_features=feature_columns)

# Convertendo numpy.ndarray para DataFrame antes de salvar
X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=encoded_columns)
y_train_resampled_df = pd.DataFrame(y_train_resampled, columns=[target_column])
X_test_processed_df = pd.DataFrame(X_test_processed, columns=encoded_columns)
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)

# Salvando os DataFrames
save_df_train_resampled = SaveDf(X_train_resampled_df)
save_df_train_resampled.save_to_csv('X_train_resampled.csv')

save_df_test_processed = SaveDf(X_test_processed_df)
save_df_test_processed.save_to_csv('X_test_processed.csv')

save_df_y_train_resampled = SaveDf(y_train_resampled_df)
save_df_y_train_resampled.save_to_csv('y_train_resampled.csv')

save_df_y_test = SaveDf(y_test_df)
save_df_y_test.save_to_csv('y_test.csv')
