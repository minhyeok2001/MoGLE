## 1. 모델 로드하기
## 2. data handling (프롬프트 합치기)
## 3. pipeline 
import wandb
import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import nest_asyncio
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LongformerForSequenceClassification, LongformerTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from langchain_groq import ChatGroq
import asyncio

from utils import inject_layerwise_lora, MultiExpertLoraLinear, capture_attention_mask

nest_asyncio.apply()

GROQ_KEY = os.environ["GROQ_API_KEY"]
STYLE_MODEL = SentenceTransformer("StyleDistance/styledistance")
EMBED_MODEL = SentenceTransformer("intfloat/multilingual-e5-large")
CLASSIFIER_MODEL_NAME = "allenai/longformer-base-4096"
GENRE_LABELS = ['Adventure', 'Dystopian', 'Fantasy', 'Horror', 'Sci-Fi']
CLASSIFIER_CKPT_PATH = "/checkpoints/longformer_classifier.ckpt"
SOTA_CENTROID_PATH = "/checkpoints/sota_centroids.pt"

class GenrePredictor:
    def __init__(self, ckpt_path, model_name, num_labels, labels_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels_list = labels_list

        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Classifier ckpt not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def get_soft_labels(self, text: str):
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        result = {}
        for idx, prob in enumerate(probs):
            label_name = (
                self.labels_list[idx] if idx < len(self.labels_list) else f"Class_{idx}"
            )
            result[label_name] = prob.item()

        return result

_GENRE_PREDICTOR = None

def get_genre_predictor():
    global _GENRE_PREDICTOR
    if _GENRE_PREDICTOR is None:
        _GENRE_PREDICTOR = GenrePredictor(
            CLASSIFIER_CKPT_PATH,
            CLASSIFIER_MODEL_NAME,
            num_labels=len(GENRE_LABELS),
            labels_list=GENRE_LABELS,
        )
    return _GENRE_PREDICTOR


SOTA1 = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.0,
    api_key=GROQ_KEY,
)
SOTA2 = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    api_key=GROQ_KEY,
)
SOTA3 = ChatGroq(
    model="moonshotai/kimi-k2-instruct",
    temperature=0.0,
    api_key=GROQ_KEY,
)

def preprocess_csv(csv_path, split_type="all"):
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


@torch.no_grad()
def generate_with_model(prompt_list, tokenizer, model, device="cuda", max_new_tokens=512):
    full_outputs = []
    model_only_outputs = []

    for p in prompt_list:
        inputs = tokenizer(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        input_ids = inputs["input_ids"]

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        # 전체 출력 (prompt + model)
        full_output = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # 모델이 새로 생성한 부분만
        generated_ids = out_ids[0][input_ids.shape[1]:]
        model_only_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        full_outputs.append(full_output)
        model_only_outputs.append(model_only_output)

    return full_outputs, model_only_outputs


### 배치처리방식
@torch.no_grad()
def generate_with_model_batched(
    prompt_list,
    tokenizer,
    model,
    device="cuda",
    batch_size=4,
    max_new_tokens=512,
):
    full_outputs = []
    model_only_outputs = []
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token_id or tokenizer.eos_token

    for i in range(0, len(prompt_list), batch_size):
        batch_prompts = prompt_list[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        input_ids = inputs["input_ids"]

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,  
        )

        for j in range(len(batch_prompts)):
            seq = out_ids[j]
            full_text = tokenizer.decode(seq, skip_special_tokens=True)

            real_input_len = (input_ids[j] != tokenizer.pad_token_id).sum().item()

            gen_ids = seq[real_input_len:]
            model_only_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            full_outputs.append(full_text)
            model_only_outputs.append(model_only_text)

    return full_outputs, model_only_outputs


# =================== SOTA COMPARISON ====================
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

def get_sota_outputs_sync(prompt):
    try:
        return asyncio.run(get_sota_outputs(prompt))
    except RuntimeError:
        return asyncio.get_event_loop().run_until_complete(get_sota_outputs(prompt))


def build_sota_centroids(prompt_lists, genre_list):
    accum = defaultdict(lambda: {"sota1": [], "sota2": [], "sota3": []})

    num_examples = len(genre_list)

    for i in range(num_examples):
        g = str(genre_list[i])

        for prompt_list in prompt_lists:
            prompt = str(prompt_list[i])
            
            sota1_out, sota2_out, sota3_out = get_sota_outputs_sync(prompt)

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
def judge_single(llm, context, user_input, response_only, genre):
    prompt = f"""
You are an evaluator for the {genre} genre.

Your job is to evaluate how well the **new reply** from the model fits the required narrative style, tone, and characteristics of the {genre} genre, **given the full previous conversation context**.

Scoring rules:
- 0.0 = completely does NOT match the genre
- 1.0 = matches the genre almost perfectly
- You MUST output only a single number between 0.0 and 1.0. No explanation.

[Genre Rules]
{context}

[Previous Conversation Context]
{user_input}

[Model's New Reply]
{response_only}
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
def genre_classifier(user_input, response_cumulative, genre):
    predictor = get_genre_predictor()
    text_for_cls = response_cumulative.strip()

    soft_labels = predictor.get_soft_labels(text_for_cls)

    pred_genre, pred_prob = max(soft_labels.items(), key=lambda x: x[1])

    true_prob = soft_labels.get(str(genre), 0.0)

    correct = 1.0 if pred_genre == str(genre) else 0.0

    return {
        "pred_genre": pred_genre,
        "correct": correct,
        "true_prob": true_prob,
        "soft_labels": soft_labels,
    }
        

# =================== EVAL PIPE ====================
def eval_pipe(prompt_list, answer_list, answer_only_list, gt_list, gt_only_list, genre_list, context_map, sota_centroids):
    """
    prompt_list: 처음에 주는 프롬프트 
    answer_list: MoLE의 output값 
    genre_list: df에서 순서대로 가져오는 genre list
    context_map: genre rules
    """
    assert len(prompt_list) == len(answer_list) == len(genre_list) == len(gt_only_list), "길이 안 맞음!!!"

    all_scores = []
    for i, (user_input, response_cumulative, response_only, gt_cumulative ,gt_only, genre) in enumerate(zip(prompt_list, answer_list, answer_only_list, gt_list,gt_only_list, genre_list)):
        print(f"\n=== Example {i} (genre={genre}) ===")
        print(f"[Prompt]\n{user_input}...")
        print(f"[Response]\n{response_only}...")
        
        if genre in context_map:
            context = context_map[genre]
        else:
            context = context_map.get("default", "")

        print("============ SOTA COMPARISON 실행중... ============")
        sota_scores = sota_comparison(response_cumulative, genre, sota_centroids)
        print("[sota_comparison_scores]", sota_scores)

        print("============ LLM JUDGE 실행중... ============")
        llm_judge_scores = llm_judge(context, user_input, response_only, genre)
        print("[llm_judge_scores]", llm_judge_scores)

        print("============ STYLE DISTANCE 실행중... ============")
        style_cos_scores = style_distance(gt_cumulative, response_cumulative)
        print("[style_distance_scores]", style_cos_scores)

        print("============ GENRE CLASSIFIER 실행중... ============")
        genre_classifier_scores = genre_classifier(user_input, response_cumulative, genre)
        print("[genre_classifier_scores]", genre_classifier_scores)
        
        all_scores.append({
            "sota_comparison_score":sota_scores,
            "llm_judge_score":llm_judge_scores,
            "style_distance_score":style_cos_scores,
            "genre_classifier_score":genre_classifier_scores
        })

    return all_scores

def summarize_scores(all_scores, title=None, prefix=""):
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
    
    cls_correct_list = [s["genre_classifier_score"]["correct"] for s in all_scores]
    cls_true_prob_list = [s["genre_classifier_score"]["true_prob"] for s in all_scores]
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    
    cls_acc = mean(cls_correct_list)
    cls_true_prob_avg = mean(cls_true_prob_list)
    
    if title:
        print(f"\n========== {title} ==========")
    else:
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

    print("\n[Style distance]")
    print(f"  style_distance avg (cosine): {mean(style_list):.4f}")
    
    print("\n[Genre classifier metrics]")
    print(f"  genre_cls_acc       : {cls_acc:.4f}")
    print(f"  genre_cls_true_prob : {cls_true_prob_avg:.4f}")

    metrics = {
        f"{prefix}num_examples": n,
        f"{prefix}sota1_cos_sim_avg": mean(sota1_list),
        f"{prefix}sota2_cos_sim_avg": mean(sota2_list),
        f"{prefix}sota3_cos_sim_avg": mean(sota3_list),
        f"{prefix}sota_avg_cos_sim": mean(sota_avg_list),
        f"{prefix}judge1_avg": mean(judge1_list),
        f"{prefix}judge2_avg": mean(judge2_list),
        f"{prefix}judge3_avg": mean(judge3_list),
        f"{prefix}judge_overall_avg": mean(judge_avg_list),
        f"{prefix}style_distance_avg": mean(style_list),
        f"{prefix}genre_cls_acc" : cls_acc,
        f"{prefix}genre_cls_true_prob" : cls_true_prob_avg
    }

    print("=====================================\n")

    wandb.log(metrics)

    return metrics


def run(args):
    
    wandb.init(
        project="MoLE_inference",
        config={
            "genre": args.genre,
            "max_new_token": args.max_new_token,
        },
    )
        
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
    if args.genre.lower() != "all":
        df = df[df["genre"] == args.genre].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"CSV 안에 genre가 '{args.genre}' 인 행이 없어~")
    
    genre_list = df["genre"].tolist()

    prompt_list_v2 = df["prompt_v2"].tolist()

    gt_list_v2 = df["GT2"].tolist()
    
    gt_cumulative_list_v2 = df["target_v2"].tolist()
    
    print("\n===== Prompt v2 (sample) =====")
    for i in range(min(3, len(prompt_list_v2))):
        print(f"[{i}] {prompt_list_v2[i]}\n")
    
    print("\n\n======== Loading precomputed SOTA centroids ========")
    SOTA_CENTROID_PATH = "/checkpoints/sota_centroids.pt"

    if not os.path.exists(SOTA_CENTROID_PATH):
        raise RuntimeError(f"SOTA centroid file not found: {SOTA_CENTROID_PATH}")

    sota_centroids = torch.load(SOTA_CENTROID_PATH, map_location="cpu")
    print("Loaded SOTA centroids!")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  
    )
    
    path = os.path.join("checkpoints",args.ft_ckpt_path)

    print(f"== Loading fine-tuned ckpt from: {path}")
    state_dict = torch.load(path, map_location="cpu")
    base_model.load_state_dict(state_dict)

    base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    base_model.register_forward_pre_hook(capture_attention_mask, with_kwargs=True)
    base_model.eval()

    max_new_tokens = args.max_new_token
    
    print("\n\n======== Running BASE (no MoLE) ========")
    answer_list_v2_base, answer_only_list_v2_base= generate_with_model_batched(prompt_list_v2, tokenizer, base_model, device=device, batch_size=args.batch_size, max_new_tokens=max_new_tokens)

    print("\n===== Llama raw outputs =====")
    for i, (prompt, only_out, gt_only, genre) in enumerate(
        zip(prompt_list_v2, answer_only_list_v2_base, gt_list_v2, genre_list)
    ):
        print(f"\n--- Example {i} / genre={genre} ---")
        print("[Prompt]")
        print(prompt)
        print("\n[GT (only)]")
        print(gt_only)
        print("\n[Llama model_only_output]")
        print(only_out)
        print("---------------")
        
        
    print("\n===== eval =====")
    scores_v2_base = eval_pipe(prompt_list_v2, answer_list_v2_base, answer_only_list_v2_base, gt_cumulative_list_v2, gt_list_v2,  genre_list, context_map, sota_centroids)
    all_base_scores = scores_v2_base
    summarize_scores(all_base_scores, title="BASE MODEL SUMMARY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", type=str, required=True)
    parser.add_argument("--max_new_token",type=int,default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ft_ckpt_path", type=str, required=True)
    args = parser.parse_args()

    run(args)