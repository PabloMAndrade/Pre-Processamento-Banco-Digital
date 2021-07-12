# Pré-Processamento de Dados

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv('/content/credit_risk_dataset.csv')

base_credit.describe()

base_credit[base_credit['person_income'] >= 6.220968e+04]

np.unique(base_credit['loan_status'], return_counts=True) #14578 pagam e 4504 devem o banco

sns.countplot(x = base_credit['loan_status']);

plt.hist(x = base_credit['person_age']);

plt.hist(x = base_credit['person_income']);

# Passando em 3 x 3 , idade da pessoa ,patrimonio , divida no banco e o color é se deve ou não ( 0 sim , 1 não ) :

grafico = px.scatter_matrix(base_credit , dimensions = ['person_age','person_income','loan_amnt'], color='loan_status')
grafico.show()

#Começando a tratar os outlines impossiveis (123 e 144 anos de idade) - Tratamento de Dados
base_credit.loc[base_credit['person_age'] > 100]

base_credit[base_credit['person_age'] > 100].index #Descobrir quais dados são "irreais" 123 e 144 anos

base_credit3 = base_credit.drop(base_credit[base_credit['person_age'] > 100].index) # Apagando os numeros "tortos" da lista

base_credit3

base_credit = base_credit3 
base_credit.head(82)

base_credit3.loc[base_credit3['person_age'] > 100]

#Apagar os dados é a ultima opcao , pois se for poucos dados o certo é corrigir as informacoes

base_credit.mean() #Calcula a média de cada coluna

base_credit['person_age'][base_credit['person_age']> 0].mean()

base_credit.head(10)

#tratamento de valores faltantes. Toda Machine Learning precisa ter os dados completos

base_credit.isnull().sum() #Verificando quais colunas estão nulas e faltam dados ex : 895 linhas na coluna person_emp_length

base_credit.loc[pd.isnull(base_credit['person_emp_length'])] # Codigo pra ver se estão nulos mesmo 

base_credit['person_emp_length'].fillna(base_credit['person_emp_length'].mean(), inplace = True)

base_credit.loc[base_credit['person_emp_length'] >= 3.213314] #Verificar se colocou a média mesmo.

base_credit['loan_int_rate'].fillna(base_credit['loan_int_rate'].mean(), inplace = True)

base_credit.loc[base_credit['loan_int_rate'] >= 10.018222] #Verificar se colocou a média mesmo.

#Usando iloc

X_credit = base_credit.iloc[:,[0,1,6]].values 
X_credit

Y_credit = base_credit.iloc[:,8].values
Y_credit

base_credit = base_credit3
base_credit.head(82) #ver se ainda tem numeros maiores que 100

X_credit

X_credit[:,0].min() , X_credit[:,1].min() , X_credit[:,2].min()  # menor idade , patrimonio e divida (0,1,2)

X_credit[:,0].max() , X_credit[:,1].max() , X_credit[:,2].max() # maior idade , patrimonio e divida (0,1,2)

from sklearn.preprocessing import StandardScaler #Uma das bibliotecas principais Machine learning
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit) # Se encaixar nos dados e fazer a transformacao (fit_transform)

X_credit[:,0].min() , X_credit[:,1].min() , X_credit[:,2].min()

X_credit

