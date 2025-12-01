## 1. 모델 로드하기
## 2. data handling (프롬프트 합치기)
## 3. pipeline 

import os
import argparse
import torch
import pandas as pd
import nest_asyncio
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from langchain_groq import ChatGroq

from utils import inject_layerwise_lora, MultiExpertLoraLinear, capture_attention_mask

nest_asyncio.apply()

GROQ_KEY = os.environ["GROQ_API_KEY"]
STYLE_MODEL = SentenceTransformer("StyleDistance/styledistance")
EMBED_MODEL = SentenceTransformer("intfloat/multilingual-e5-large")

SOTA1 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    api_key=GROQ_KEY,
)
SOTA2 = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.0,
    api_key=GROQ_KEY,
)
SOTA3 = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=GROQ_KEY,
)

def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)

    def build_prompts(row):
        u1 = str(row["사람1"])
        a1 = str(row["AI1"])
        u2 = str(row["사람2"])
        a2 = str(row["AI2"])
        u3 = str(row["사람3"])

        prompt_v1 = f"user: {u1}"

        prompt_v2 = "\n".join([
            f"user: {u1}",
            f"assistant: {a1}",
            f"user: {u2}",
        ])

        prompt_v3 = "\n".join([
            f"user: {u1}",
            f"assistant: {a1}",
            f"user: {u2}",
            f"assistant: {a2}",
            f"user: {u3}",
        ])

        return pd.Series({
            "prompt_v1": prompt_v1,
            "prompt_v2": prompt_v2,
            "prompt_v3": prompt_v3,
        })

    prompts = df.apply(build_prompts, axis=1)
    df = pd.concat([df, prompts], axis=1)
    return df

@torch.no_grad()
def generate_with_model(prompt_list, tokenizer, model, device="cuda", max_new_tokens=512):
    outputs = []
    for p in prompt_list:
        inputs = tokenizer(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs


# =================== SOTA COMPARISON ====================
def build_sota_centroids(prompt_lists, genre_list):
    accum = defaultdict(lambda: {"sota1": [], "sota2": [], "sota3": []})

    num_examples = len(genre_list)

    for i in range(num_examples):
        g = str(genre_list[i])

        for prompt_list in prompt_lists:
            prompt = str(prompt_list[i])
            
            sota1_out = SOTA1.invoke(prompt).content
            sota2_out = SOTA2.invoke(prompt).content
            sota3_out = SOTA3.invoke(prompt).content

            embs = EMBED_MODEL.encode(
                [sota1_out, sota2_out, sota3_out],
                convert_to_tensor=True,
            )
            e1, e2, e3 = embs[0], embs[1], embs[2]

            accum[g]["sota1"].append(e1)
            accum[g]["sota2"].append(e2)
            accum[g]["sota3"].append(e3)

    centroids = {}
    for g, d in accum.items():
        sota1_cent = torch.stack(d["sota1"], dim=0).mean(dim=0)
        sota2_cent = torch.stack(d["sota2"], dim=0).mean(dim=0)
        sota3_cent = torch.stack(d["sota3"], dim=0).mean(dim=0)
        avg_cent   = torch.stack([sota1_cent, sota2_cent, sota3_cent], dim=0).mean(dim=0)

        centroids[g] = {
            "sota1": sota1_cent,
            "sota2": sota2_cent,
            "sota3": sota3_cent,
            "avg":   avg_cent,
        }

    return centroids

def sota_comparison(response, genre, sota_centroids):
    g = str(genre)
    if g not in sota_centroids:
        return {
            "sota1_cos_sim": 0.0,
            "sota2_cos_sim": 0.0,
            "sota3_cos_sim": 0.0,
            "avg_cos_sim":   0.0,
        }

    cent = sota_centroids[g]

    resp_emb = EMBED_MODEL.encode(response, convert_to_tensor=True)

    s1 = cos_sim(resp_emb, cent["sota1"]).item()
    s2 = cos_sim(resp_emb, cent["sota2"]).item()
    s3 = cos_sim(resp_emb, cent["sota3"]).item()
    avg = cos_sim(resp_emb, cent["avg"]).item()

    return {
        "sota1_cos_sim": s1,
        "sota2_cos_sim": s2,
        "sota3_cos_sim": s3,
        "avg_cos_sim":   avg,
    }
    
    
# =================== LLM JUDGE ====================
def judge_single(llm, context, user_input, response, genre):
    prompt = f"""
    You are a {genre} genre evaluator.
    Based on the [Genre Rules] below, evaluate how well the [Model Output] follows the required style, tone, and narrative characteristics.

    0.0 = does not match at all
    1.0 = matches almost perfectly

    You must output only a single number between 0.0 and 1.0. Do not include any explanation.

    [Genre Rules]
    {context}

    [User Prompt]
    {user_input}

    [Model Output]
    {response}
    """.strip()

    result = llm.invoke(prompt)
    score_str = result.content.strip()

    try:
        return float(score_str)
    except:
        return 0.0


def llm_judge(context, user_input, response, genre):
    base_scores = {
        "judge1": judge_single(SOTA1, context, user_input, response, genre),
        "judge2": judge_single(SOTA2, context, user_input, response, genre),
        "judge3": judge_single(SOTA3, context, user_input, response, genre),
    }
    avg = sum(base_scores.values()) / len(base_scores)

    scores = {**base_scores, "avg": avg}
    return scores

# =================== STYLE DISTANCE ====================
def style_distance(gt, response):
    embeddings = STYLE_MODEL.encode(
        [gt, response],
        convert_to_tensor=True
    )
    sim = cos_sim(embeddings[0], embeddings[1]).item()
    return sim
    
# =================== GENRE CLASSIFIER ====================
def genre_classifier(user_input, response, genre):
    pass
    
    
# =================== EVAL PIPE ====================
def eval_pipe(prompt_list, answer_list, gt_list, genre_list, context_map, sota_centroids):
    """
    prompt_list: 처음에 주는 프롬프트 
    answer_list: MoLE의 output값 
    genre_list: df에서 순서대로 가져오는 genre list
    context_map: genre rules
    """
    assert len(prompt_list) == len(answer_list) == len(genre_list) == len(gt_list), "길이 안 맞음!!!"

    all_scores = []
    for i, (user_input, response, gt, genre) in enumerate(zip(prompt_list, answer_list, gt_list, genre_list)):
        print(f"\n=== Example {i} (genre={genre}) ===")
        print(f"[Prompt]\n{user_input[:120]}...")
        print(f"[Response]\n{response[:120]}...")
        
        if genre in context_map:
            context = context_map[genre]
        else:
            context = context_map.get("default", "")

        print("============ SOTA COMPARISON 실행중... ============")
        sota_scores = sota_comparison(response, genre, sota_centroids)
        print("[sota_comparison_scores]", sota_scores)

        print("============ LLM JUDGE 실행중... ============")
        llm_judge_scores = llm_judge(context, user_input, response, genre)
        print("[llm_judge_scores]", llm_judge_scores)

        print("============ STYLE DISTANCE 실행중... ============")
        style_cos_scores = style_distance(gt, response)
        print("[style_distance_scores]", style_cos_scores)

        print("============ GENRE CLASSIFIER 실행중... ============")
        #genre_classifier_scores = genre_classifier(user_input, response, genre)
        #print("[genre_classifier_scores]", genre_classifier_scores)
        
        all_scores.append({
            "index": i,
            "genre": genre,
            "prompt": user_input,
            "response": response,
            "sota_comparison_score":sota_scores,
            "llm_judge_score":llm_judge_scores,
            "style_distance_score":style_cos_scores,
            #"genre_classifier_score":genre_classifier_scores
        })

    return all_scores

def summarize_scores(all_scores, title=None):
    n = len(all_scores)

    sota1_list = [s["sota_comparison_score"]["sota1_cos_sim"] for s in all_scores]
    sota2_list = [s["sota_comparison_score"]["sota2_cos_sim"] for s in all_scores]
    sota3_list = [s["sota_comparison_score"]["sota3_cos_sim"] for s in all_scores]
    sota_avg_list = [s["sota_comparison_score"]["avg_cos_sim"] for s in all_scores]

    judge1_list = [s["llm_judge_score"]["judge1"] for s in all_scores]
    judge2_list = [s["llm_judge_score"]["judge2"] for s in all_scores]
    judge3_list = [s["llm_judge_score"]["judge3"] for s in all_scores]
    judge_avg_list = [s["llm_judge_score"]["avg"] for s in all_scores]

    style_list = [s["style_distance_score"] for s in all_scores]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n========== OVERALL SUMMARY ==========")
    print(f"#examples: {n}")

    print("\n[SOTA cosine similarity]")
    print(f"  sota1_cos_sim avg: {mean(sota1_list):.4f}")
    print(f"  sota2_cos_sim avg: {mean(sota2_list):.4f}")
    print(f"  sota3_cos_sim avg: {mean(sota3_list):.4f}")
    print(f"  sota_avg_cos_sim : {mean(sota_avg_list):.4f}")

    print("\n[LLM Judge scores]")
    print(f"  judge1 avg: {mean(judge1_list):.4f}")
    print(f"  judge2 avg: {mean(judge2_list):.4f}")
    print(f"  judge3 avg: {mean(judge3_list):.4f}")
    print(f"  judge overall avg: {mean(judge_avg_list):.4f}")

    print("\n[Style distance (GT vs MoLE)]")
    print(f"  style_distance avg (cosine): {mean(style_list):.4f}")
    print("=====================================\n")
    

def run(args):
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    LORA_BASE_PATH = "/checkpoints"
    GATE_CKPT_PATH = f"/checkpoints/gate_ckpts/mole_{args.gate_weight}.ckpt"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = "cuda"
    context_map = {
        "Adventure": """
        - Fast or steady forward-moving pacing
        - Clear sense of journey, exploration, or mission
        - Action-oriented scenes or problem-solving moments
        - Atmosphere emphasizing challenge, thrill, or discovery
        """.strip(),

        "Horror": """
        - Dark, tense, or unsettling atmosphere
        - Gradual build-up of fear, dread, or anxiety
        - Elements of threat, mystery, or the unknown
        - Emotional tone that evokes discomfort or suspense
        """.strip(),

        "Fantasy": """
        - Fantasy setting with magic or supernatural elements
        - Consistent worldbuilding and internal logic
        - Emotional but not melodramatic tone
        - Characters, events, or visuals reflecting a mythical or otherworldly feel
        """.strip(),

        "Sci-Fi": """
        - Technology, science, or futuristic concepts integrated into the narrative
        - Logical or speculative worldbuilding
        - Analytical or reflective tone rather than purely emotional
        - Themes involving innovation, artificial intelligence, space, or advanced society
        """.strip(),

        "Dystopian": """
        - Bleak, oppressive, or controlled societal structure
        - Themes of surveillance, inequality, or loss of freedom
        - Dark, reflective emotional tone
        - Protagonist perspective highlighting resistance, suffering, or systemic issues
        """.strip(),

        "default": "General narrative quality.",
    }

    df = preprocess_csv("eval.csv")
    genre_list = df["genre"].tolist()

    prompt_list_v1 = df["prompt_v1"].tolist()
    prompt_list_v2 = df["prompt_v2"].tolist()
    prompt_list_v3 = df["prompt_v3"].tolist()

    gt_list_v1 = df["GT1_gemini"].tolist()
    gt_list_v2 = df["GT2_gemini"].tolist()
    gt_list_v3 = df["GT3_gemini"].tolist()
    
    print("\n===== Prompt v1 (sample) =====")
    for i in range(min(3, len(prompt_list_v1))):
        print(f"[{i}] {prompt_list_v1[i]}\n")

    print("\n===== Prompt v2 (sample) =====")
    for i in range(min(3, len(prompt_list_v2))):
        print(f"[{i}] {prompt_list_v2[i]}\n")

    print("\n===== Prompt v3 (sample) =====")
    for i in range(min(3, len(prompt_list_v3))):
        print(f"[{i}] {prompt_list_v3[i]}\n")
    
    print("\n\n======== Building SOTA centroids (by genre, using v1+v2+v3) ========")
    sota_centroids = build_sota_centroids(
        [prompt_list_v1, prompt_list_v2, prompt_list_v3],
        genre_list,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    base_model.register_forward_pre_hook(capture_attention_mask, with_kwargs=True)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    print("\n\n======== Running BASE Llama-3.1-8B (no MoLE) ========")
    answer_list_v1_base = generate_with_model(prompt_list_v1, tokenizer, base_model, device=device, max_new_tokens=512)
    answer_list_v2_base = generate_with_model(prompt_list_v2, tokenizer, base_model, device=device, max_new_tokens=512)
    answer_list_v3_base = generate_with_model(prompt_list_v3, tokenizer, base_model, device=device, max_new_tokens=512)

    scores_v1_base = eval_pipe(prompt_list_v1, answer_list_v1_base, gt_list_v1, genre_list, context_map, sota_centroids)
    scores_v2_base = eval_pipe(prompt_list_v2, answer_list_v2_base, gt_list_v2, genre_list, context_map, sota_centroids)
    scores_v3_base = eval_pipe(prompt_list_v3, answer_list_v3_base, gt_list_v3, genre_list, context_map, sota_centroids)

    all_base_scores = scores_v1_base + scores_v2_base + scores_v3_base
    summarize_scores(all_base_scores, title="BASE MODEL SUMMARY (Llama-3.1-8B-Instruct)")

    print("\n\n======== Injecting MoLE and evaluating ========")

    expert_files = [
        f for f in os.listdir(LORA_BASE_PATH)
        if f.startswith("expert_") and f.endswith(".ckpt")
    ]

    def extract_genre(fname):
        return fname[len("expert_") : -len(".ckpt")]

    GENRES = sorted(extract_genre(f) for f in expert_files)
    print("genres:", GENRES)

    expert_ckpt_paths = [
        os.path.join(LORA_BASE_PATH, f"expert_{g}.ckpt") for g in GENRES
    ]

    model_mole = inject_layerwise_lora(
        base_model,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"],
        num_experts=len(GENRES),
        r=8,
        lora_alpha=16,
        expert_ckpt_paths=expert_ckpt_paths,
    )

    gate_sd = torch.load(GATE_CKPT_PATH, map_location="cpu")
    model_mole.load_state_dict(gate_sd, strict=False)
    model_mole.to(device)

    model_mole.eval()
    for p in model_mole.parameters():
        p.requires_grad = False

    answer_list_v1_mole = generate_with_model(prompt_list_v1, tokenizer, model_mole, device=device, max_new_tokens=512)
    answer_list_v2_mole = generate_with_model(prompt_list_v2, tokenizer, model_mole, device=device, max_new_tokens=512)
    answer_list_v3_mole = generate_with_model(prompt_list_v3, tokenizer, model_mole, device=device, max_new_tokens=512)

    scores_v1_mole = eval_pipe(prompt_list_v1, answer_list_v1_mole, gt_list_v1, genre_list, context_map, sota_centroids)
    scores_v2_mole = eval_pipe(prompt_list_v2, answer_list_v2_mole, gt_list_v2, genre_list, context_map, sota_centroids)
    scores_v3_mole = eval_pipe(prompt_list_v3, answer_list_v3_mole, gt_list_v3, genre_list, context_map, sota_centroids)

    all_mole_scores = scores_v1_mole + scores_v2_mole + scores_v3_mole
    summarize_scores(all_mole_scores, title="MoLE MODEL SUMMARY (Llama-3.1-8B + MoLE)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate_weight", type=str, required=True)
    args = parser.parse_args()

    run(args)