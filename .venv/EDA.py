import pandas as pd
import matplotlib.pyplot as plt
import random
from classes_eda import DropColumn,Nulls,InfoDf,Age_Eng,Unic,Analise_Graf_Dist,SaveDf


# Lendo o DataFrame
df = pd.read_csv('C:/Users/Gdper/health_care/.venv/healthcare_dataset.csv')
pd.set_option('display.max_columns', None)

# Removendo múltiplas colunas
dropper = DropColumn(df)
df = dropper.drop_col(['Name', 'Doctor', 'Room Number', 'Date of Admission', 'Discharge Date', 'Hospital',
                       'Billing Amount'])

# Informacoes sobre o df
info = InfoDf(df)
info.info_df()

# Verificando dados Nulos no Df

null_checker = Nulls(df)
null_checker.is_null()

# Verificando dados Unicos 
unic_checker = Unic(df)
unique_counts = unic_checker.unique_data()
print(unique_counts)

# Engenharia de atributo na variavel idade
age_engineer = Age_Eng(df)
df = age_engineer.age_eng()

# Excluindo variavel Age apos Engenharia de atributo
df = dropper.drop_col('Age')

# Analise Grafica da Distribuicao dos dados Categoricos
analise = Analise_Graf_Dist(df)
analise.bar_plot()
plt.show()

# salvando Df pos analise exploratoria como novo df
save_df = SaveDf(df)
save_df.save_to_csv('df_pos_eda.csv')