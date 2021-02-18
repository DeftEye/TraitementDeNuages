import pandas as pd

import sys
sys.path.append()

label_map = {
        'Fish': 0,
        'Flower': 1,
        'Gravel': 2,
        'Sugar': 3
        }

df_train = pd.read_csv(train)
df_train['Image'] = df_train.Image_Label.map(lambda v: v[:v.find('_')])
df_train['Label'] = df_train.Image_Label.map(lambda v: v[v.find('_')+1:])
df_train['LabelIndex'] = df_train.Label.map(lambda v: label_map[v])