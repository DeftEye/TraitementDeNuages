
import pandas as pd



classes = {
    'Fish': 0,
    'Flower': 1,
    'Gravel': 2,
    'Sugar': 3
}

# Dataset train

csv_train_path = 'test_3.csv'

df_train = pd.read_csv(csv_train_path)
print(df_train)
print(df_train.columns)
df_train.drop(df_train[df_train['Presence']==0].index, inplace = True)
df_train = df_train.groupby(by ='Image', axis = 0)['Largeur'].apply(list)
print(df_train)
# df_train['bbox'] = [df_train['Largeur']] + [df_train['Taille']] + [df_train['cx']] + [df_train['cy']]
# print(df_train)