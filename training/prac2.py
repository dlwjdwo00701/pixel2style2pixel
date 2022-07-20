import pandas as pd
from deepface import DeepFace

mask = list(pd.read_csv('True_data.csv')['input'])
result = []

for i in mask:
    result.append(DeepFace.verify(img1_path=))
