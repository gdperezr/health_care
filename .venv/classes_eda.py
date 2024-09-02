import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


##### Classes para Analises Exploratorias #####


# Deleta coluna do DF

class DropColumn:
    def __init__(self, df):
        self.df = df

    def drop_col(self, columns):

        self.df = self.df.drop(columns=columns)
        return self.df


# Verificar Dados Nulos no Df
class Nulls:
    def __init__(self, df):
        self.df = df

    def is_null(self):
        # Calcula a soma dos valores nulos por coluna
        nulls = self.df.isnull().sum()
        print(nulls)
        return nulls
    
# Describe e Info sobre o Df
class InfoDf:

    def __init__(self,df):
        self.df = df

    def info_df(self):
        print("Informações do DataFrame:")
        self.df.info()
        print("\nEstatísticas Descritivas:")
        print(self.df.describe())

# Feature Eng Idade
class Age_Eng:

    def __init__(self, df):
        self.df = df

    def age_eng(self):
        self.df['faixa_etaria'] = pd.cut(self.df['Age'],
                                         bins=[0, 18, 35, 60, 100],
                                         labels=['Criança', 'Jovem', 'Adulto', 'Idoso'])
        return self.df


#  Contagem Valores Unicos por coluna
class Unic:

    def __init__(self, df):
        self.df = df

    def unique_data(self):
        print('Verifica o número de valores únicos em cada coluna')
        unique_counts = self.df.nunique()
        print(unique_counts)
        return unique_counts

# Separacao dos dados Categoricos e Grafico de Barras para analisar Distribuicao
class Analise_Graf_Dist:

    def __init__(self, df):
        self.df = df

    def bar_plot(self):
        categorical = []

        for col in self.df.columns:
            if self.df[col].dtype == 'object' or self.df[col].dtype == 'category':
                categorical.append(col)


        for cat in categorical:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.df[cat])
            plt.title(f'Distribution of {cat}')
            

# Salvando como novo Df pos analise exploratoria
class SaveDf:
    def __init__(self, df):
        self.df = df

    def save_to_csv(self, filename):
        self.df.to_csv(filename, index=False)
        print(f"DataFrame salvo como {filename}")
