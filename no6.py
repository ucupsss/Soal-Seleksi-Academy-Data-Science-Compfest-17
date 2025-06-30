import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")

weather_cols = train.drop(columns=["ID", "date", "cluster_id", "electricity_consumption"]).columns

corr_matrix = train[weather_cols].corr()

corr_pairs = (corr_matrix.abs()
              .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
              .stack()
              .reset_index()
              .rename(columns={"level_0":"Variable 1",
                               "level_1":"Variable 2",
                               0:"|Pearson r|"})
                               .sort_values("|Pearson r|", ascending=False))

print(corr_pairs[corr_pairs["|Pearson r|"] > 0.80])
