import pandas as pd
from sklearn import svm

pd.set_option('display.max_columns',None)

X_train_resampled = pd.read_csv('C:/Users/Gdper/health_care/.venv/X_train_resampled.csv')
y_train_resampled = pd.read_csv('C:/Users/Gdper/health_care/.venv/y_train_resampled.csv')

X_test_processed = pd.read_csv('C:/Users/Gdper/health_care/.venv/X_test_processed.csv')
y_test = pd.read_csv('C:/Users/Gdper/health_care/.venv/y_test.csv')

# Convertendo y_train_resampled e y_test para Series (unidimensional)
y_train_resampled = y_train_resampled.squeeze()  # Remove o eixo extra, se presente
y_test = y_test.squeeze()  # Remove o eixo extra, se presente

# Treinando o modelo SVM
clf = svm.SVC()
clf.fit(X_train_resampled, y_train_resampled)

# Avaliando o modelo
accuracy = clf.score(X_test_processed, y_test)
print(f"Accuracy: {accuracy:.4f}")
