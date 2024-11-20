import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("../data/raw/admission.csv")
df = df.drop(['Serial No.'], axis=1)
df = df.rename(columns={
    "GRE Score": "GRE_Score",
    "TOEFL Score": "TOEFL_Score",
    "University Rating": "University_Rating",
    "Chance of Admit ": "Chance_of_Admit"
})

X = df.drop('Chance_of_Admit', axis=1)
y = df['Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=789)

os.makedirs('../data/processed', exist_ok=True)
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)

print("Les fichiers CSV ont été enregistrés dans le dossier 'data/processed'.")
