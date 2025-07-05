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

df_big5 = pd.read_csv("./dataset/dataset/survey/BigFive.csv")
df_grades = pd.read_csv("./dataset/dataset/education/grades.csv")
df_stress = pd.read_csv("./dataset/dataset/survey/PerceivedStressScale.csv")
df_flourishing_scale = pd.read_csv("./dataset/dataset/survey/FlourishingScale.csv")
df_PHQ = pd.read_csv("./dataset/dataset/survey/PHQ-9.csv")
df_psqi=pd.read_csv("./dataset/dataset/survey/psqi.csv")
df_vr = pd.read_csv("./dataset/dataset/survey/vr_12.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

profiles = []
uids = []
profile_texts = []

for uid in df_big5['uid'].unique():
    try:
        big5 = createChunkFromDf(df_big5, uid)
        gpa = createChunkFromDf(df_grades, uid)
        stress = createChunkFromDf(df_stress, uid)
        flourishing_scale = createChunkFromDf(df_flourishing_scale, uid)
        text = [
            f"Student {uid}:\n"
        ]

        text = text + big5 + gpa + stress + flourishing_scale + PHQ + psqi + vr
        uids.append(uid)
        profile_texts.append('\n'.join(text))
        PHQ = createChunkFromDf(df_PHQ, uid)
        psqi = createChunkFromDf(df_psqi, uid)
        vr = createChunkFromDf(df_vr, uid)

        text = [
            f"Student {uid}:\n"
        ]

        text = text + big5 + gpa + stress + flourishing_scale + PHQ + psqi + vr
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
        {"uid": uid, "profile": text}
        for uid, text in zip(uids, profile_texts)
    ], f, indent=2)

print(f"Stored {len(uids)} student profiles in FAISS.")