import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

def createChunkFromDf(df: pd.DataFrame, uid: str):
    try:
        dictionary = df[df['uid'] == uid].iloc[0]
        return [f'{key}: {value}' for key, value in dictionary.items() if key != 'uid']
    except:
        print(f"Skipping uid {uid} for dataframe {df.head()}")
        return []

df_data = pd.read_csv("student_data_extended - Training.csv")


model = SentenceTransformer("all-MiniLM-L6-v2")

profiles = []
uids = []
profile_texts = []

for uid in df_data['uid'].unique():
    try:
        data = createChunkFromDf(df_data, uid)
        text = [
            f"Student {uid}:\n"
        ]

        text = text + data
        uids.append(uid)
        profile_texts.append('\n'.join(text))

    except Exception as e:
        print(f"Skipping {uid}: {e}")

embeddings = model.encode(profile_texts, convert_to_numpy=True).astype('float32')

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "studentlife_profiles.faiss")

with open("studentlife_profiles.json", "w") as f:
    json.dump([
        {"uid": int(uid), "profile": text}
        for uid, text in zip(uids, profile_texts)
    ], f, indent=2)


print(f"Stored {len(uids)} student profiles in FAISS.")
