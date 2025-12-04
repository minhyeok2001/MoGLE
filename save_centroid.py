import os
import torch
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import asyncio
from langchain_groq import ChatGroq

# =========================
# CONFIG
# =========================
GROQ_KEY = os.environ["GROQ_API_KEY"]
SAVE_PATH = "/checkpoints/sota_centroids.pt"   # 저장 파일

STYLE_MODEL = SentenceTransformer("StyleDistance/styledistance")
EMBED_MODEL = SentenceTransformer("intfloat/multilingual-e5-large")

# SOTA MODELS
SOTA1 = ChatGroq(model="qwen/qwen3-32b", temperature=0.0, api_key=GROQ_KEY)
SOTA2 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0, api_key=GROQ_KEY)
SOTA3 = ChatGroq(model="moonshotai/kimi-k2-instruct", temperature=0.0, api_key=GROQ_KEY)


# =========================
# ASYNC CALL
# =========================
async def get_sota_outputs(prompt):
    results = await asyncio.gather(
        SOTA1.ainvoke(prompt),
        SOTA2.ainvoke(prompt),
        SOTA3.ainvoke(prompt),
        return_exceptions=True
    )

    outs = []
    for r in results:
        outs.append(r.content)
    return outs


def get_sota_sync(prompt):
    try:
        return asyncio.run(get_sota_outputs(prompt))
    except RuntimeError:
        return asyncio.get_event_loop().run_until_complete(get_sota_outputs(prompt))


# =========================
# BUILD CENTROIDS
# =========================
def build_sota_centroids(df):
    prompt_list = df["prompt_v2"].tolist()
    genre_list = df["genre"].tolist()

    accum = defaultdict(lambda: {"s1": [], "s2": [], "s3": []})

    print("=== Running SOTA models per sample ===")
    for i in range(len(df)):
        g = str(genre_list[i])
        prompt = str(prompt_list[i])

        print(f"[{i}/{len(df)}] genre={g}")

        s1, s2, s3 = get_sota_sync(prompt)

        embs = EMBED_MODEL.encode([s1, s2, s3], convert_to_tensor=True)
        e1, e2, e3 = embs[0], embs[1], embs[2]

        accum[g]["s1"].append(e1)
        accum[g]["s2"].append(e2)
        accum[g]["s3"].append(e3)

    centroids = {}
    for g, d in accum.items():
        c1 = torch.stack(d["s1"], dim=0).mean(dim=0)
        c2 = torch.stack(d["s2"], dim=0).mean(dim=0)
        c3 = torch.stack(d["s3"], dim=0).mean(dim=0)
        avg = torch.stack([c1, c2, c3], dim=0).mean(dim=0)

        centroids[g] = {
            "sota1": c1,
            "sota2": c2,
            "sota3": c3,
            "avg":   avg,
        }

    return centroids



def preprocess_csv(csv_path, split_type):
    df = pd.read_csv(csv_path)

    n = len(df)
    if split_type == "A":
        df = df.iloc[: n // 2].reset_index(drop=True)
    elif split_type == "B":
        df = df.iloc[n // 2 :].reset_index(drop=True)
    elif split_type =="all":
        df = df
    else:
        raise RuntimeError("유효한 TYPE 주세요")


    def build_prompts(row):
        u1 = str(row["사람1"])
        a1 = str(row["AI1"])
        u2 = str(row["사람2"])
        a2 = str(row["AI2"])
        u3 = str(row["사람3"])

        GT1 = str(row["GT1"])
        GT2 = str(row["GT2"])
        GT3 = str(row["GT3"])

        prompt_v1 = f"user: {u1}"
        target_v1 = GT1

        prompt_v2 = "\n".join([
            f"user: {u1}",
            f"assistant: {a1}",
            f"user: {u2}",
        ])
        target_v2 = prompt_v2 + "\nassistant: " + GT2

        prompt_v3 = "\n".join([
            f"user: {u1}",
            f"assistant: {a1}",
            f"user: {u2}",
            f"assistant: {a2}",
            f"user: {u3}",
        ])
        target_v3 = prompt_v3 + "\nassistant: " + GT3

        return pd.Series({
            "prompt_v1": prompt_v1,
            "prompt_v2": prompt_v2,
            "prompt_v3": prompt_v3,
            "target_v1": target_v1,
            "target_v2": target_v2,
            "target_v3": target_v3,
        })

    prompts = df.apply(build_prompts, axis=1)
    df = pd.concat([df, prompts], axis=1)
    return df

def main():
    df = pd.read_csv("eval.csv")
    df = preprocess_csv("eval.csv","all")
    print("=== Building SOTA centroids ===")
    centroids = build_sota_centroids(df)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(centroids, SAVE_PATH)
    print(f"\nSOTA centroids saved to:\n  {SAVE_PATH}")


if __name__ == "__main__":
    main()