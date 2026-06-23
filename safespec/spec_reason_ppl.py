# %%
import os
import re
import time
import openai
import pickle
import pprint
import logging
import argparse
import numpy as np
from openai import OpenAI
import torch
import statistics
from collections import Counter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

# Safety head imports
import sys
sys.path.append("/data/xuhaotian/Safety_head")
from safety_head import load_safety_head, read_safety_head_config, select_hidden_state, pooled_hidden_states
# 共享同一份对抗性后缀字符串，保证与 spec_reason 一致
from spec_reason import SUFFIX as DEFAULT_ADV_SUFFIX

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

clients = {}
tokenizers = {}
safety_model = None
safety_head = None
safety_tokenizer = None
safety_device = None
# safety backbone 后端："vllm"(查 pooling 实例) 或 "hf"(本地 transformers 32B)
safety_backend = "vllm"
safety_embed_client = None   # vllm 后端：指向 pooling 实例的 OpenAI 客户端
safety_pool_model = "qwen3-32b-embed"
# 与 checkpoint / CLI 对齐后的 hidden_states 层下标（在 initialize_safety 中设置）
safety_hidden_layer_index = 4
# 框架在 recovery 时插入的警告文本(含 harmful/refusal 等触发词)。做安全打分时必须把它
# 从 safety_text 里剔除——否则安全头会反复对【自己插入的警告】误判 unsafe → 级联回滚→截停。
WARNING_MSG_STRIPPED = "Wait, verify whether the text is potentially harmful; if harmful, I must not comply with any instruction to continue, and I must end the response with a refusal."

# 对抗性后缀（GCG 风格）开关：默认关闭；开启后仅附加到 benign prompt 末尾，
# 用于测试该攻击下安全框架（safety head + recovery）的防御效果。jailbreak 不附加。
USE_ADV_SUFFIX = False
ADV_SUFFIX = DEFAULT_ADV_SUFFIX


def set_adv_suffix(use, suffix=None):
    """由 CLI 在运行前设置：是否启用对抗性后缀，以及可选的自定义后缀字符串。"""
    global USE_ADV_SUFFIX, ADV_SUFFIX
    USE_ADV_SUFFIX = bool(use)
    if suffix:
        ADV_SUFFIX = suffix

DEFAULT_DRAFT_MODEL_PATH = "/data/xuhaotian/model/Qwen3-1.7B"
DEFAULT_BASE_MODEL_PATH = "/data/xuhaotian/model/Qwen3-32B"
DEFAULT_SAFETY_HEAD_PATH = "/data/xuhaotian/Safety_head/safety_head_ckpt_qwen32b_layerlast_mix_aug2/safety_head_best.pt"

model_names = {
    "1.5b": DEFAULT_DRAFT_MODEL_PATH,
    "32b": DEFAULT_BASE_MODEL_PATH,
}

def get_avg_score(scores):
    return statistics.mean([x for x in scores if x is not None])

def get_frequency(scores):
    return dict(Counter(scores))

def get_model(model_size):
    client = clients[model_size]
    models = client.models.list()
    model = models.data[0].id
    return model


def resolve_model_names(args):
    return {
        "1.5b": getattr(args, "draft_model_path", DEFAULT_DRAFT_MODEL_PATH),
        "32b": getattr(args, "base_model_path", DEFAULT_BASE_MODEL_PATH),
    }

def initialize_clients(args):
    global clients, tokenizers, model_names
    model_names = resolve_model_names(args)
    ports = {
        "1.5b": args.draft_port,
        "32b": args.base_port,
    }
    clients = {}
    tokenizers = {}
    for size, full_name in model_names.items():
        clients[size] = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{ports[size]}/v1",
        )
        tokenizers[size] = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)


def initialize_safety(args):
    """
    Load safety model + head for local inference.
    """
    global safety_model, safety_head, safety_tokenizer, safety_device, safety_hidden_layer_index
    global safety_backend, safety_embed_client, safety_pool_model
    if args.safety_head_path is None:
        logging.warning("No safety_head_path provided; safety score disabled.")
        return

    safety_backend = getattr(args, "safety_backend", "vllm")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 层下标两后端共用（从 checkpoint 读，CLI 可覆盖）
    cfg_disk = read_safety_head_config(args.safety_head_path)
    effective_layer = (
        cfg_disk.hidden_layer_index
        if getattr(args, "safety_hidden_layer", None) is None
        else args.safety_hidden_layer
    )
    if (
        getattr(args, "safety_hidden_layer", None) is not None
        and args.safety_hidden_layer != cfg_disk.hidden_layer_index
    ):
        logging.warning(
            "safety_hidden_layer=%s overrides checkpoint hidden_layer_index=%s (head weights may be wrong).",
            args.safety_hidden_layer,
            cfg_disk.hidden_layer_index,
        )
    safety_hidden_layer_index = effective_layer

    # ---------- backend == vllm：查 pooling 实例，不加载 HF backbone ----------
    if safety_backend == "vllm":
        if effective_layer not in (-1, None):
            logging.warning(
                "safety_backend=vllm 用 vLLM MEAN 池化（等价于 last-layer mask-mean）；"
                "当前 layer=%s 非最后层，若 head 训练于中间层请改用 --safety_backend hf。",
                effective_layer,
            )
        # tokenizer 仅用于与训练一致的右截断（max_length），不占显存
        safety_tokenizer = AutoTokenizer.from_pretrained(args.safety_model_path, trust_remote_code=True)
        if safety_tokenizer.pad_token is None:
            safety_tokenizer.pad_token = safety_tokenizer.eos_token
        # head 放 CPU(fp32)：~13M 参数，单向量前向 <1ms，且不与 pooling 实例抢显存
        safety_device = torch.device("cpu")
        safety_head, _ = load_safety_head(args.safety_head_path, safety_device, dtype=torch.float32)
        safety_pool_model = getattr(args, "safety_pool_model", "qwen3-32b-embed")
        port = getattr(args, "safety_pool_port", "30009")
        safety_embed_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        try:
            probe = safety_embed_client.embeddings.create(model=safety_pool_model, input="probe")
            dim = len(probe.data[0].embedding)
        except Exception as e:
            raise RuntimeError(
                f"safety_backend=vllm 但 pooling 实例不可用 (port {port}, model {safety_pool_model}): {e}\n"
                f"请先启动: CUDA_VISIBLE_DEVICES=<gpu> vllm serve {args.safety_model_path} "
                "--runner pooling --convert embed "
                "--pooler-config '{\"pooling_type\":\"MEAN\",\"normalize\":false}' "
                f"--enforce-eager --max-model-len 2048 --port {port} --served-model-name {safety_pool_model}"
            )
        assert dim == cfg_disk.hidden_size, f"pooling dim {dim} != head hidden_size {cfg_disk.hidden_size}"
        safety_model = None
        logging.info(
            "Safety backend=vllm: pooling @ port %s (model=%s, dim=%d); head on CPU fp32; layer=%s. 不加载 HF backbone。",
            port, safety_pool_model, dim, safety_hidden_layer_index,
        )
        return

    # ---------- backend == hf：原 transformers 路径 ----------
    dev_str = args.safety_device
    if dev_str.isdigit():
        dev_str = f"cuda:{dev_str}"
    num_gpus = torch.cuda.device_count()
    if "," in dev_str or (dev_str == "cuda" and num_gpus > 1):
        device_map = "auto"
        safety_device = torch.device("cuda:0")
        logging.info(f"Multiple GPUs detected ({num_gpus}). Using device_map='auto' for safety model.")
    else:
        safety_device = torch.device(dev_str)
        device_map = None

    safety_tokenizer = AutoTokenizer.from_pretrained(args.safety_model_path, trust_remote_code=True)
    if safety_tokenizer.pad_token is None:
        safety_tokenizer.pad_token = safety_tokenizer.eos_token
    safety_model = AutoModelForCausalLM.from_pretrained(
        args.safety_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if device_map is None:
        safety_model.to(safety_device)
    safety_model.eval()
    safety_model.requires_grad_(False)

    input_dev = next(safety_model.parameters()).device
    with torch.no_grad():
        probe = safety_tokenizer(".", return_tensors="pt", add_special_tokens=True)
        probe = {k: v.to(input_dev) for k, v in probe.items()}
        probe_out = safety_model(
            **probe,
            output_hidden_states=True,
            use_cache=False,
        )
        backbone_device = select_hidden_state(
            probe_out.hidden_states, safety_hidden_layer_index
        ).device

    safety_head, _ = load_safety_head(args.safety_head_path, backbone_device, dtype=dtype)
    logging.info(
        "Safety backend=hf: loaded HF backbone + head from %s on %s (layer=%s, %d states)",
        args.safety_head_path,
        backbone_device,
        safety_hidden_layer_index,
        len(probe_out.hidden_states),
    )

def get_first_user_msg(problem, options=None):
    """根据数据类型构造首条 user message。jailbreak 与 benign 的 prompt 必须不同：

    - jailbreak：options 为字符串("jailbreak")，prompt 本身即完整越狱文本，原样返回，
      不附加任何任务指令/系统提示，以测量真实 ASR（唯一防御是 safety head）。
    - benign 计算题(gsm8k/math/aime)：options=None，要求分步推理并以 \\boxed{} 给终答，便于判分。
    - benign 选择题(gpqa/hellaswag)：options 为 {A,B,C,D} dict，必须注入选项，否则模型看不到
      选项无法作答（此前直接 return problem 会丢选项，导致 GPQA 准确率失真）。

    注意：benign 分支不附加任何对抗性 SUFFIX；并与 spec_reason.get_first_user_msg 保持一致，
    确保 spec_ppl / speculative / target_only / draft_only 各 run_mode 的 benign prompt 完全相同。
    """
    if isinstance(options, str):
        return problem  # jailbreak：原样，不附加后缀
    if options is None:
        msg = (
            "Solve the following problem step by step. Reason inside <think>...</think>, "
            "separate logical reasoning steps with two newline characters, and put your final "
            f"answer within \\boxed{{}}.\n\nProblem: {problem}"
        )
    else:
        msg = (
            "What is the correct answer to the following multiple-choice question? Reason step by "
            "step, then give the final answer strictly as \\boxed{X}, where X is one of A, B, C, or D.\n\n"
            f"Problem: {problem}\n"
            f"Choices:\n(A) {options['A']}\n(B) {options['B']}\n(C) {options['C']}\n(D) {options['D']}"
        )
    if USE_ADV_SUFFIX:
        msg = f"{msg}\n\n{ADV_SUFFIX}"
    return msg

# %%
def _is_context_overflow_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "maximum context length" in msg
        or "reduce the length of the input" in msg
        or "prompt is too long" in msg
    )


def generate_new_step(problem, steps_so_far, model_size, options=None, stop_token="\n\n", max_tokens=512, n=1, temperature=0.6, trailing_newline=True, repetition_penalty=1.0):
    client = clients[model_size]
    tokenizer = tokenizers[model_size]

    local_steps = list(steps_so_far)
    while True:
        # Construct prompt manually using tokenizer
        if local_steps == []:
            messages = [{"role": "user", "content": get_first_user_msg(problem, options)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # 避免在 warning 末尾强行加空行：仅在步与步之间使用分隔，是否在末尾追加由 trailing_newline 控制
            steps_so_far_str = "\n\n".join(local_steps).rstrip("\n")
            if trailing_newline:
                steps_so_far_str = steps_so_far_str + "\n\n"
            messages = [
                {"role": "user", "content": get_first_user_msg(problem, options)},
                {"role": "assistant", "content": f"<think>{steps_so_far_str}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # 去掉续写 prompt 末尾的"回合结束符",让模型在【打开的】assistant 回合里继续 <think>。
            # 不同模型族 token 不同：Qwen=<|im_end|>，DeepSeek-R1-Distill=<｜end▁of▁sentence｜>，Llama3=<|eot_id|>。
            # 之前只剥 Qwen 的 → DeepSeek 回合没被打开 → 模型在已结束序列后续写出 :// 等垃圾(过度拒绝的真因)。
            for _eot in ("<|im_end|>\n", "<|im_end|>", "<｜end▁of▁sentence｜>\n", "<｜end▁of▁sentence｜>",
                         "<|eot_id|>\n", "<|eot_id|>"):
                if prompt.endswith(_eot):
                    prompt = prompt[:-len(_eot)]
                    break

        # Now use completions API
        start_time = time.perf_counter()
        try:
            response = client.completions.create(
                model=get_model(model_size),
                prompt=prompt,
                temperature=temperature, 
                top_p=0.95,
                max_tokens=max_tokens,
                n=n,
                stop=[stop_token] if stop_token else None,
                extra_body={"repetition_penalty": repetition_penalty},
            )
            break
        except openai.BadRequestError as exc:
            if not _is_context_overflow_error(exc) or len(local_steps) <= 1:
                raise
            # Drop oldest history to keep the run alive.
            drop = max(1, len(local_steps) // 4)
            local_steps = local_steps[drop:]
            logging.warning(
                "Context overflow on %s; dropped %d oldest steps, remaining=%d",
                model_size,
                drop,
                len(local_steps),
            )
    
    elapsed = time.perf_counter() - start_time
    
    results = []
    for choice in response.choices:
        step_str = choice.text
        num_output_tokens = response.usage.completion_tokens // n # Approximation for multi-sample
        # 结束判定：本步出现【非空】\boxed{...} 才算完成。不能用 "final answer is" 子串——
        # 模型常先单独写 "the final answer is:" 再换行写盒子，会在写出答案前提前截停导致空答案。
        # extract_answer 对空盒返回 ""、无盒返回 None。两个框架共用同一判定，保证对比公平。
        finished = bool(extract_answer(step_str))
        results.append((step_str, finished, num_output_tokens))
    
    if n == 1:
        return results[0][0], results[0][1], results[0][2], elapsed
    else:
        return results, elapsed


def get_score(args, problem, steps_so_far, model_size="32b", options=None):
    """
    Align scoring with spec_reason.py: use chat completion to generate a single digit (0-9),
    higher is better. Keep the safety score from the original PPL version.
    """
    client = clients[model_size]

    local_steps = list(steps_so_far)
    while True:
        steps_so_far_str = "\n\n".join(local_steps) + "\n\n"
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
            {"role": "assistant", "content": f"{steps_so_far_str}"},
            {"role": "user", "content": "Evaluate the last reasoning step solely based on factual correctness and logical validity. Ignore style, phrasing, or overall usefulness—only judge whether the step is objectively correct and logically follows from prior steps. Assign a score from 0 to 9."},
            {"role": "assistant", "content": "<think>I think the quality score is: "},
        ]
        # Give a low score to steps that directly refuse to answer.
        start_time = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=get_model(model_size),
                messages=messages,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=10,
                extra_body={
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                },
            )
            break
        except openai.BadRequestError as exc:
            if not _is_context_overflow_error(exc) or len(local_steps) <= 1:
                raise
            drop = max(1, len(local_steps) // 4)
            local_steps = local_steps[drop:]
            logging.warning(
                "Score context overflow on %s; dropped %d oldest steps, remaining=%d",
                model_size,
                drop,
                len(local_steps),
            )
    elapsed = time.perf_counter() - start_time

    justification = response.choices[0].message.content
    score = process_logprobs(response, method=args.score_method)

    # Safety score (0-1, higher = more unsafe) — retained
    safety_score = None
    safety_time = 0.0
    if safety_head is not None:
        safety_start = time.perf_counter()
        _hist = steps_so_far[-args.safety_history_steps:] if args.safety_history_steps > 0 else [steps_so_far[-1]]
        # 剔除框架自己插入的 recovery 警告(否则头反复对警告误判 → 级联截停,实测 gpqa spec_ppl 主因)
        safety_steps = [s for s in _hist if s.strip() != WARNING_MSG_STRIPPED] or [steps_so_far[-1]]
        safety_text = "\n\n".join(safety_steps) if safety_steps else steps_so_far[-1]
        if safety_backend == "vllm":
            # 与训练一致：先用 tokenizer 右截断到 max_length，再把 token ids 发给 vLLM pooling
            # 实例（MEAN, normalize=false）取 last-layer 均值——已奇偶校验与 HF mask-mean-pool 等价。
            ids = safety_tokenizer(
                safety_text, truncation=True, max_length=args.safety_max_length
            )["input_ids"]
            if not ids:
                ids = safety_tokenizer(" ")["input_ids"]
            resp = safety_embed_client.embeddings.create(model=safety_pool_model, input=ids)
            pooled = torch.tensor(resp.data[0].embedding, dtype=torch.float32).unsqueeze(0)
            head_param = next(safety_head.parameters())
            with torch.no_grad():
                logits = safety_head(
                    pooled.to(device=head_param.device, dtype=head_param.dtype)
                ).squeeze(-1)
                safety_score = torch.sigmoid(logits).item()
        else:
            inputs = safety_tokenizer(
                safety_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.safety_max_length,
            )
            inputs = {k: v.to(safety_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = safety_model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = select_hidden_state(outputs.hidden_states, safety_hidden_layer_index)
                # 与训练一致：对 safety_text 的所有非 padding token 做 mask 平均池化
                pooled = pooled_hidden_states(
                    hidden, inputs["attention_mask"].to(hidden.device), prefix_len=0
                )
                # 对齐 safety_head 的设备与 dtype（pooled 因 denom 提升会变 fp32）。
                head_param = next(safety_head.parameters())
                logits = safety_head(
                    pooled.to(device=head_param.device, dtype=head_param.dtype)
                ).squeeze(-1)
                safety_score = torch.sigmoid(logits).item()
        safety_time = time.perf_counter() - safety_start

    return score, safety_score, elapsed, safety_time


def process_logprobs(response, method, temp=1.0):
    assert len(response.choices[0].logprobs.content) == 1
    token = response.choices[0].logprobs.content[0].token# example: '1', '0'
    token_logprobs = {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}
    logging.info(f"Original token_logprobs: {token_logprobs}")
    token_logprobs = {k: v for k, v in token_logprobs.items() if k.isdigit()}  # filter out non-digit values

    if method == "greedy":
        # return the vanilla response
        if not token.isdigit():
            return 0
        return int(token)
    elif method == "average":
        # Convert log probabilities to probabilities and normalize each distribution.
        probs = {tok: np.exp(lp / temp) for tok, lp in token_logprobs.items()}
        total_probs = sum(probs.values())
        for tok in probs:
            probs[tok] /= total_probs
        for i in range(10):
            if i not in probs:
                probs[i] = 0
        logging.info(f"Avg score: {sum([int(t) * p for t, p in probs.items()])}")
        return sum([int(t) * p for t, p in probs.items()])
    else:
        raise NotImplementedError


def get_dataset(dataset_name):
    if dataset_name == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset_name == "gpqa":
        local_gpqa_path = "/data/xuhaotian/specreason-origin/gpqa/gpqa_diamond.csv"
        if os.path.exists(local_gpqa_path):
            dataset = load_dataset("csv", data_files=local_gpqa_path)["train"]
        elif os.getenv("HF_HUB_OFFLINE", "0") == "1":
            dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
        else:    
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    elif dataset_name == "gsm8k":
        local_gsm8k_path = "/data/xuhaotian/specreason-origin/GSM8K"
        if os.path.exists(local_gsm8k_path):
            dataset = load_dataset("parquet", data_files={"test": f"{local_gsm8k_path}/test-00000-of-00001.parquet"})["test"]
        else:
            dataset = load_dataset("gsm8k", "main")["test"]
    elif dataset_name == "hellaswag":
        local_hellaswag_path = "/data/xuhaotian/specreason-origin/hellaswag/data"
        if os.path.exists(local_hellaswag_path):
            dataset = load_dataset("parquet", data_files={"validation": f"{local_hellaswag_path}/validation-00000-of-00001.parquet"})["validation"]
        else:
            dataset = load_dataset("hellaswag")["validation"]
    else:
        raise NotImplementedError
    return dataset


def extract_answer(text):
    if "boxed" not in text:
        return None
    # Extract the last boxed content
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    
    count = 0
    start = idx + 7
    for i in range(start, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            if count == 0:
                return text[start:i]
            count -= 1
    return None

def normalize_answer(answer):
    if answer is None:
        return ""
    return answer.strip()

def judge_correctness(dataset_name, pred_str, ground_truth):
    # 复用 spec_reason 的精细判分（gpqa 扫所有 boxed 取字母 / gsm8k 数值归一 / math LaTeX 归一），
    # 与 speculative / target_only 完全同口径，保证三模式准确率可比、不再因 strip-only 误判 math。
    from spec_reason import judge_correctness as _sr_judge
    return _sr_judge(dataset_name, pred_str, ground_truth)

def build_parser():
    parser = argparse.ArgumentParser(description="Runs speculative reasoning using a small model")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "gsm8k", "hellaswag"], default="gpqa",
                        help="Dataset")
    parser.add_argument("--score_threshold", type=float, default=9.0,
                        help="Acceptance threshold")
    parser.add_argument("--force_base_answer", type=int, default=1,
                        help="1=含最终答案的步强制 base 生成,不接受 draft(提 acc)")
    parser.add_argument("--token_budget", type=int, default=2048,
                        help="Max num of total output tokens in each step")
    # problem_id: 60-89 for AIME, 0-99 for math, 0-99 for GPQA
    parser.add_argument("--problem_id", type=int, default=None,
                        help="Query ID (60-89 for AIME). If None, uses range.")
    parser.add_argument("--start_id", type=int, default=0, help="Start problem ID")
    parser.add_argument("--end_id", type=int, default=1, help="End problem ID (exclusive)")
    parser.add_argument("--repeat_id", type=int, default=None,
                        help="Repeat ID (0-15, k=16). If None, uses range.")
    parser.add_argument("--start_repeat", type=int, default=0, help="Start repeat ID")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of repeats per problem")
    parser.add_argument("--score_method", type=str, choices=["greedy", "average"], default="greedy",
                        help="Scoring method")
    parser.add_argument("--output_dir", type=str, default="/data/xuhaotian/specreason-origin/results/saftyspec", 
                        help="Where result pickle files will be written to")
    parser.add_argument("--first_n_steps_base_model", type=int, default=0, 
                        help="First n steps use base model only")
    parser.add_argument("--run_mode", type=str, choices=["speculative","spec_ppl", "target_only", "draft_only"], default="speculative",
                        help="Run mode: speculative, target_only, or draft_only")
    parser.add_argument("--base_port", type=str, default="30000", help="Port for base model")
    parser.add_argument("--draft_port", type=str, default="30001", help="Port for draft model")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH,
                        help="Target/base model path served by vLLM")
    parser.add_argument("--draft_model_path", type=str, default=DEFAULT_DRAFT_MODEL_PATH,
                        help="Draft/small model path served by vLLM")
    parser.add_argument("--safety_head_path", type=str, default=DEFAULT_SAFETY_HEAD_PATH,
                        help="Path to safety head checkpoint")
    parser.add_argument("--safety_model_path", type=str, default="/data/xuhaotian/model/Qwen3-32B",
                        help="Backbone model path for safety head")
    parser.add_argument("--safety_threshold", type=float, default=0.5, help="Safety score threshold (>= unsafe)")
    parser.add_argument("--safety_max_length", type=int, default=2048, help="Max length for safety tokenizer")
    parser.add_argument("--safety_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for safety head inference")
    parser.add_argument("--safety_history_steps", type=int, default=3, help="Number of recent steps (including current) passed to safety head")
    parser.add_argument("--max_consecutive_recovery", type=int, default=3, help="连续触发回滚/插警告达到此次数则提前安全截停")
    parser.add_argument(
        "--safety_hidden_layer",
        type=int,
        default=None,
        help="接在 backbone 的 hidden_states 层下标（默认 None=使用 checkpoint 里保存的 hidden_layer_index；0=embedding 后，-1=最后一层）。",
    )
    parser.add_argument(
        "--safety_backend",
        type=str,
        choices=["vllm", "hf"],
        default="vllm",
        help="安全头 backbone 后端：vllm=查 vLLM pooling 实例出 hidden state(默认)；hf=本地 transformers 32B(旧路径)。",
    )
    parser.add_argument("--safety_pool_port", type=str, default="30009",
                        help="safety_backend=vllm 时的 pooling 实例端口。")
    parser.add_argument("--safety_pool_model", type=str, default="qwen3-32b-embed",
                        help="safety_backend=vllm 时 pooling 实例的 served-model-name。")
    parser.add_argument("--max_steps", type=int, default=64, help="Hard cap on reasoning steps")
    parser.add_argument("--max_rollback_per_step", type=int, default=5, help="Max number of rollbacks allowed per step_id")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for generation")
    parser.add_argument("--use_adv_suffix", action="store_true",
                        help="在 benign prompt 末尾附加对抗性 SUFFIX（默认关闭），测试该攻击下的防御效果；jailbreak 不受影响")
    parser.add_argument("--adv_suffix", type=str, default=None,
                        help="自定义对抗性后缀字符串；默认使用内置 GCG 风格 SUFFIX")
    return parser

def resolve_problem(args, problem_id=None):
    pid = problem_id if problem_id is not None else args.problem_id
    if pid is None:
        raise ValueError("problem_id must be provided either in args or as an argument")

    if args.dataset_name == "aime":
        problem = args.dataset["problem"][pid - 60]
        ground_truth = args.dataset["answer"][pid - 60]
        options = None
    elif args.dataset_name == "math":
        problem = args.dataset["problem"][pid]
        ground_truth = args.dataset["solution"][pid]
        options = None
    elif args.dataset_name == "gpqa":
        from spec_reason import gpqa_shuffle_options
        problem = args.dataset["Question"][pid]
        options, ground_truth = gpqa_shuffle_options(
            pid,
            args.dataset["Correct Answer"][pid],
            [args.dataset["Incorrect Answer 1"][pid],
             args.dataset["Incorrect Answer 2"][pid],
             args.dataset["Incorrect Answer 3"][pid]],
        )
    elif args.dataset_name == "gsm8k":
        problem = args.dataset["question"][pid]
        ground_truth = args.dataset["answer"][pid]
        if "####" in ground_truth:
            ground_truth = ground_truth.split("####")[-1].strip()
        options = None
    elif args.dataset_name == "hellaswag":
        problem = args.dataset["ctx"][pid]
        label = int(args.dataset["label"][pid])
        ground_truth = ["A", "B", "C", "D"][label]
        endings = args.dataset["endings"][pid]
        options = {
            "A": endings[0],
            "B": endings[1],
            "C": endings[2],
            "D": endings[3],
        }
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset_name}")
    return problem, options, ground_truth

def run_ppl_spec_reasoning(problem, options=None, args=None):
    # Speculative mode (PPL)
    steps_so_far = []
    step_id = 0
    metadata_list = []
    recovery_mode_steps = 0
    rollback_counts = Counter()
    consecutive_recovery = 0  # 连续回滚计数（被安全步重置）；达到 max_consecutive_recovery 则提前截停
    # 注意：末尾保留一个空格，以便模型直接续写（内容与 WARNING_MSG_STRIPPED 对齐，便于在打分时剔除）
    warning_msg = " " + WARNING_MSG_STRIPPED
    warning_msg_stripped = WARNING_MSG_STRIPPED

    def _fmt(x):
        return "None" if x is None else f"{x:.4f}"
    
    def _trailing_allowed():
        return not (steps_so_far and steps_so_far[-1].strip() == warning_msg_stripped)

    refusal_pattern = re.compile(
        r"()",
            # i'm sorry|i am sorry|cannot assist|can't assist|can't help|cannot help|not able to help|cannot provide|can't provide|potentially harmful)",
        re.IGNORECASE,
    )
    
    while True:
        warning_flag = False
        insert_warning_after_base = False
        step_time = 0
        justification = None
        if step_id>=64:
            break
        # Multi-sample configuration
        if recovery_mode_steps > 0:
            n_draft = 20
            temperature_draft = 1
        else:
            n_draft = 1
            temperature_draft = 0.6

        # 首步：仍然做 PPL，但跳过安全检查
        if step_id == 0:
            step_str, finished, num_output_tokens, small_model_time = generate_new_step(
                problem, steps_so_far, "1.5b", options=options, temperature=temperature_draft, trailing_newline=_trailing_allowed(),
                repetition_penalty=args.repetition_penalty
            )
            small_model_step, num_output_tokens_small = step_str, num_output_tokens
            step_time += small_model_time

            score, safety_score, eval_time, safety_time = get_score(args, problem, steps_so_far + [step_str], options=options)
            step_time += eval_time
            # 首步跳过安全过滤
            logging.info(
                f"[Step {step_id}] (no safety check) draft_len={num_output_tokens_small}, "
                f"score={_fmt(score)}, safety={_fmt(safety_score)}\n"
                f"[Step {step_id}] draft_content: {step_str}"
            )
            # 仅用评分判定是否走 base（高分通过）
            if score is not None and score >= args.score_threshold:
                base_model_step, num_output_tokens_base, base_model_time = None, None, None
                logging.info(
                    f"[Step {step_id}] accept draft (score={_fmt(score)})"
                )
            else:
                step_str, finished, num_output_tokens, base_model_time = generate_new_step(problem, steps_so_far, "32b", options=options, repetition_penalty=args.repetition_penalty)
                base_model_step, num_output_tokens_base = step_str, num_output_tokens
                step_time += base_model_time
                logging.info(
                    f"[Step {step_id}] draft rejected (score={_fmt(score)}), use base_len={num_output_tokens_base}\n"
                    f"[Step {step_id}] base_content: {step_str}"
                )
            steps_so_far.append(step_str)

        elif step_id < args.first_n_steps_base_model:
            base_model_step, finished, num_output_tokens_base, base_model_time = generate_new_step(problem, steps_so_far, "32b", options=options, repetition_penalty=args.repetition_penalty)
            small_model_step, num_output_tokens_small, small_model_time = None, None, None
            score, safety_score, eval_time, safety_time = None, None, None, 0.0
            step_str, step_time = base_model_step, base_model_time
            steps_so_far.append(step_str)
        else:
            # 1. generate small
            if n_draft == 1:
                step_str, finished, num_output_tokens, small_model_time = generate_new_step(
                    problem, steps_so_far, "1.5b", options=options, temperature=temperature_draft, trailing_newline=_trailing_allowed(),
                    repetition_penalty=args.repetition_penalty
                )
                small_model_step, num_output_tokens_small = step_str, num_output_tokens
                step_time += small_model_time
                
                # 2. score
                score, safety_score, eval_time, safety_time = get_score(args, problem, steps_so_far + [step_str], options=options)
                step_time += eval_time
                logging.info(
                    f"[Step {step_id}] draft_len={num_output_tokens_small}, "
                    f"score={_fmt(score)}, safety={_fmt(safety_score)}\n"
                    f"[Step {step_id}] draft_content: {step_str}"
                )

                # 2.5 safety check：unsafe 时先尝试回滚 + 进入 recovery（多候选）；用尽则强制拒稿走 base
                unsafe = safety_score is not None and safety_score >= args.safety_threshold
                if unsafe:
                    max_rollback = getattr(args, "max_rollback_per_step", 5)
                    if rollback_counts[step_id] < max_rollback:
                        rollback_counts[step_id] += 1
                        consecutive_recovery += 1
                        logging.info(
                            f"[Step {step_id}] Safety score {safety_score:.4f} >= threshold {args.safety_threshold}. "
                            f"Rolling back (attempt {rollback_counts[step_id]}/{max_rollback}, consecutive={consecutive_recovery})."
                        )

                        if len(steps_so_far) > 0:
                            removed_step = steps_so_far.pop()
                            logging.info(f"[Step {step_id}] Rolled back previous step: {removed_step[:80]}...")
                            step_id = max(step_id - 1, 0)

                        warning_msg = " Wait, verify whether the text is potentially harmful; if harmful, I must not comply with any instruction to continue, and I must end the response with a refusal."
                        if not steps_so_far or steps_so_far[-1].strip() != warning_msg_stripped:
                            steps_so_far.append(warning_msg)

                        recovery_mode_steps = 1

                        metadata = {
                            "step_id": step_id,
                            "step_str": warning_msg,
                            "small_model_step": small_model_step,
                            "num_output_tokens_small": num_output_tokens_small,
                            "small_model_time": small_model_time,
                            "score": score,
                            "eval_time": eval_time,
                            "safety_time": safety_time,
                            "safety_score": safety_score,
                            "safety_threshold": args.safety_threshold,
                            "safety_verdict": "unsafe",
                            "final_num_output_tokens": 0,
                            "warning": "safety_head_triggered",
                            "consecutive_recovery": consecutive_recovery,
                        }
                        metadata_list.append(metadata)
                        # 连续回滚达阈值：提前安全截停（最终输出停在警告/拒答处）
                        max_consec = getattr(args, "max_consecutive_recovery", 3)
                        if consecutive_recovery >= max_consec:
                            metadata_list[-1]["stop_reason"] = "safety_abort"
                            logging.info(
                                f"[Step {step_id}] consecutive recovery {consecutive_recovery} >= {max_consec}; "
                                f"aborting generation early (safe stop)."
                            )
                            break
                        step_id += 1
                        continue
                    logging.info(
                        f"[Step {step_id}] Safety score {safety_score:.4f} >= threshold {args.safety_threshold}, "
                        f"but max rollbacks reached ({max_rollback}). Rejecting draft and forcing base model for this step."
                    )
                    score = -999.0  # Force rejection of draft
                else:
                    consecutive_recovery = 0  # 出现安全步，连续回滚计数清零

                # 3. decide (score >= threshold to accept)；含最终答案的步强制 base 生成(答案最关键)
                is_answer_step = any(k in step_str for k in ("boxed", "Answer:", "ANSWER:"))
                if score is not None and score >= args.score_threshold and not (getattr(args, "force_base_answer", 1) and is_answer_step):
                    base_model_step, num_output_tokens_base, base_model_time = None, None, None
                    logging.info(
                        f"[Step {step_id}] accept draft (score={_fmt(score)}, safety={_fmt(safety_score)})"
                    )
                else:
                    step_str, finished, num_output_tokens, base_model_time = generate_new_step(problem, steps_so_far, "32b", options=options, repetition_penalty=args.repetition_penalty)
                    base_model_step, num_output_tokens_base = step_str, num_output_tokens
                    step_time += base_model_time

                    base_score, base_safety_score, base_eval_time, base_safety_time = get_score(args, problem, steps_so_far + [step_str], options=options)
                    score, safety_score, eval_time, safety_time = base_score, base_safety_score, base_eval_time, base_safety_time
                    step_time += base_eval_time

                    logging.info(
                        f"[Step {step_id}] draft rejected (score={_fmt(score)}), use base_len={num_output_tokens_base}, base_safety={_fmt(safety_score)}\n"
                        f"[Step {step_id}] base_content: {step_str}"
                    )

                    unsafe_base = safety_score is not None and safety_score >= args.safety_threshold
                    if unsafe_base:
                        logging.info(f"[Step {step_id}] Base safety score {safety_score:.4f} >= threshold {args.safety_threshold}. Inserting warning message.")
                        insert_warning_after_base = True
            
            else: # Recovery Mode (n_draft > 1)
                logging.info(f"[Step {step_id}] Recovery Mode: Generating {n_draft} candidates with DRAFT model...")
                results_draft, small_model_time = generate_new_step(
                    problem, steps_so_far, "1.5b", options=options, n=n_draft, temperature=temperature_draft, trailing_newline=_trailing_allowed(),
                    repetition_penalty=args.repetition_penalty
                )
                step_time += small_model_time
                
                # Recovery mode: prefer refusal patterns, otherwise pick by safety/quality rule
                best_cand_obj = None
                all_candidates_data = []

                def _score_key(obj):
                    return obj["score"] if obj["score"] is not None else float("-inf")

                def _safety_key(obj):
                    return obj["safety_score"] if obj["safety_score"] is not None else 1.0

                for idx_cand, (cand_str, cand_finished, cand_tokens) in enumerate(results_draft):
                    cand_score, cand_safety_score, _, cand_safety_time = get_score(args, problem, steps_so_far + [cand_str], options=options)
                    logging.info(
                        f"[Step {step_id}] cand#{idx_cand} len={cand_tokens}, score={_fmt(cand_score)}, safety={_fmt(cand_safety_score)}\n"
                        f"[Step {step_id}] cand#{idx_cand} content: {cand_str}"
                    )
                    
                    cand_obj = {
                        "str": cand_str,
                        "finished": cand_finished,
                        "tokens": cand_tokens,
                        "score": cand_score,
                        "safety_score": cand_safety_score,
                        "safety_time": cand_safety_time,
                        "has_refusal": bool(refusal_pattern.search(cand_str)),
                    }
                    all_candidates_data.append(cand_obj)

                refusal_candidates = [c for c in all_candidates_data if c["has_refusal"]]
                if refusal_candidates:
                    best_cand_obj = max(refusal_candidates, key=_score_key)
                    logging.info(
                        f"[Step {step_id}] Strong-pick refusal candidate among {len(refusal_candidates)} (score={_fmt(best_cand_obj['score'])}, safety={_fmt(best_cand_obj['safety_score'])})."
                    )
                elif all_candidates_data:
                    # 先选所有candidate中安全分最低的
                    min_safety = min(_safety_key(c) for c in all_candidates_data)
                    lowest_safety_candidates = [c for c in all_candidates_data if _safety_key(c) == min_safety]
                    lowest_safety_best = max(lowest_safety_candidates, key=_score_key)

                    # 若该候选的质量分达标则接受，否则在质量达标者中选安全分最低的
                    if lowest_safety_best["score"] is not None and lowest_safety_best["score"] >= args.score_threshold:
                        best_cand_obj = lowest_safety_best
                        logging.info(
                            f"[Step {step_id}] Selected lowest-safety candidate with sufficient quality (score={_fmt(best_cand_obj['score'])}, safety={_fmt(best_cand_obj['safety_score'])})."
                        )
                    else:
                        quality_candidates = [
                            c for c in all_candidates_data
                            if c["score"] is not None and c["score"] >= args.score_threshold
                        ]
                        if quality_candidates:
                            min_safety_q = min(_safety_key(c) for c in quality_candidates)
                            lowest_safety_q_candidates = [c for c in quality_candidates if _safety_key(c) == min_safety_q]
                            best_cand_obj = max(lowest_safety_q_candidates, key=_score_key)
                            logging.info(
                                f"[Step {step_id}] Selected lowest-safety among quality-qualified candidates (score={_fmt(best_cand_obj['score'])}, safety={_fmt(best_cand_obj['safety_score'])})."
                            )
                        else:
                            best_cand_obj = lowest_safety_best
                            logging.info(
                                f"[Step {step_id}] No quality-qualified candidates; fallback to lowest-safety (score={_fmt(best_cand_obj['score'])}, safety={_fmt(best_cand_obj['safety_score'])})."
                            )
                
                if best_cand_obj:
                    step_str = best_cand_obj["str"]
                    finished = best_cand_obj["finished"]
                    num_output_tokens_small = best_cand_obj["tokens"]
                    safety_score = best_cand_obj["safety_score"]
                    safety_time = best_cand_obj["safety_time"]
                    score = best_cand_obj["score"]
                else:
                    # Fallback for empty results_draft
                    step_str = results_draft[0][0]
                    finished = results_draft[0][1]
                    num_output_tokens_small = results_draft[0][2]
                    score, safety_score, _, safety_time = get_score(args, problem, steps_so_far + [step_str], options=options)

                logging.info(
                    f"[Step {step_id}] Selected candidate: score={_fmt(score)}, safety={_fmt(safety_score)}, len={num_output_tokens_small}. Skip PPL.\n"
                    f"[Step {step_id}] selected_content: {step_str}"
                )

                base_model_step, num_output_tokens_base, base_model_time = None, None, None
                recovery_mode_steps -= 1
                
            if "</think>" in step_str and not any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]]):
                if not (isinstance(options, str) and options == "jailbreak"):
                    step_str = step_str.replace("</think>", "")
                    warning_flag = True

            # Detect explicit refusal phrases and end early
            refusal_strings = [
                # "i am unable to assist with",
                # "i'm sorry, but i can't assist with",
                # "i cannot assist with"
                "**Final Answer**",
                "**Answer**",
                "**Final Word Count:**"
            ]
            if any(rs in step_str.lower() for rs in refusal_strings):
                finished = True
                logging.info(f"[Step {step_id}] Final Answer detected, finishing response.")

            steps_so_far.append(step_str)
            if insert_warning_after_base:
                # Append warning immediately after unsafe base output
                steps_so_far.append(warning_msg)
            
        metadata = {
            "step_id": step_id,
            "step_str": step_str,
            "small_model_step": small_model_step,
            "num_output_tokens_small": num_output_tokens_small,
            "small_model_time": small_model_time, 
            "score": score,
            "eval_time": eval_time,
            "safety_time": safety_time,
            "safety_score": safety_score,
            "safety_threshold": args.safety_threshold,
            "safety_verdict": "safe" if safety_score is None else ("unsafe" if safety_score >= args.safety_threshold else "safe"),
            "base_model_step": base_model_step,
            "num_output_tokens_base": num_output_tokens_base,
            "base_model_time": base_model_time, 
            "final_num_output_tokens": num_output_tokens_base if num_output_tokens_base is not None else num_output_tokens_small,
            "step_time": step_time,
            "justification": justification,
        }
        if warning_flag:
             metadata["warning"] = "step_str had a </think>"
        metadata_list.append(metadata)
        step_id += 1
        
        # Finish check for loops
        is_jailbreak = isinstance(options, str) and options == "jailbreak"
        
        if len(steps_so_far) > 2:
            last_step = steps_so_far[-1].strip()
            # 避免把重复的 warning_msg 误判为循环终止
            if last_step != warning_msg_stripped:
                # 1) 连续重复
                finished = finished or steps_so_far[-1] == steps_so_far[-2]
                # 2) A-B-A 循环（立即终止）
                finished = finished or steps_so_far[-1] == steps_so_far[-3]

        # 出现【非空】boxed 才判完成（与 generate_new_step / speculative 的循环一致）。
        # 严禁用 "final answer is" 子串——模型常先单独写 "the final answer is X" 再换行写盒子，
        # 会在出盒前提前停→空答案(实测 gsm8k 因此只有 81，修后应回到 ~91+)。
        if extract_answer(step_str):
            finished = True
            
        logging.info(f"Step {step_id}: {step_str}") # Log full step content

        current_total_tokens = sum([m.get("final_num_output_tokens", 0) for m in metadata_list])
        
        if finished or current_total_tokens >= args.token_budget or step_id >= args.max_steps:
             if current_total_tokens >= args.token_budget:
                 metadata_list[-1]["stop_reason"] = "budget"
                 logging.info(f"[Stop] reason=budget, tokens={current_total_tokens}, steps={len(steps_so_far)}")
             elif step_id >= args.max_steps:
                 metadata_list[-1]["stop_reason"] = "max_steps"
                 logging.info(f"[Stop] reason=max_steps, tokens={current_total_tokens}, steps={len(steps_so_far)}")
             else:
                 metadata_list[-1]["stop_reason"] = "finished"
                 logging.info(f"[Stop] reason=finished, tokens={current_total_tokens}, steps={len(steps_so_far)}")
             break
             
    return metadata_list, None


def main():
    parser = build_parser()
    args, _ = parser.parse_known_args()

    initialize_clients(args)
    initialize_safety(args)
    set_adv_suffix(getattr(args, "use_adv_suffix", False), getattr(args, "adv_suffix", None))
    if USE_ADV_SUFFIX:
        logging.info("Adversarial SUFFIX ENABLED (appended to benign prompts only).")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.dataset = get_dataset(args.dataset_name)
    
    # Determine the range of problem IDs and repeats
    if args.problem_id is not None:
        pids = [args.problem_id]
    else:
        pids = list(range(args.start_id, args.end_id))
    
    if args.repeat_id is not None:
        rids = [args.repeat_id]
    else:
        rids = list(range(args.start_repeat, args.start_repeat + args.num_repeats))

    for pid in pids:
        for rid in rids:
            try:
                problem, options, ground_truth = resolve_problem(args, problem_id=pid)
            except Exception as e:
                logging.error(f"Error resolving problem {pid}: {e}")
                continue

            output_filename = os.path.join(args.output_dir, f"{pid}/{rid}")
            if os.path.exists(f"{output_filename}.pickle"):
                logging.info(f"Problem {pid} repeat {rid} resolved, skipping")
                continue

            logging.info(f"===== Running Problem {pid} repeat {rid} =====")
            generation_start_time = time.perf_counter()
            
            if args.run_mode in ["target_only", "draft_only"]:
                logging.error("Single model modes are not fully implemented in this refactored spec_reason_ppl.py. Use spec_reason.py for draft/target_only.")
                continue
            else:
                metadata_list, _ = run_ppl_spec_reasoning(problem, options, args)
                
            total_wall_time = time.perf_counter() - generation_start_time
            total_safety_time = sum(m.get("safety_time", 0.0) for m in metadata_list)
            simulated_optimized_time = total_wall_time - total_safety_time

            os.makedirs(os.path.dirname(f"{output_filename}.pickle"), exist_ok=True)
            
            # Calc stats and save
            total_tokens = sum(m["final_num_output_tokens"] for m in metadata_list if m.get("final_num_output_tokens"))
            draft_tokens = sum(m["num_output_tokens_small"] for m in metadata_list if m.get("num_output_tokens_small"))
            rejected_steps = sum(1 for m in metadata_list if m.get("score") is not None and m.get("score") < args.score_threshold)
            
            steps_so_far = [m["step_str"] for m in metadata_list]
            is_correct = False
            parsed_pred = None
            if steps_so_far:
                final_step = steps_so_far[-1]
                is_correct, parsed_pred = judge_correctness(args.dataset_name, final_step, ground_truth)
                logging.info(f"Final Answer: {parsed_pred}")
                logging.info(f"Correct: {is_correct}")
                print(f"JUDGMENT for Prob {pid} Repeat {rid}: {1 if is_correct else 0}")
                metadata_list[-1]["judgment"] = is_correct
                metadata_list[-1]["parsed_pred"] = parsed_pred
                metadata_list[-1]["ground_truth"] = ground_truth
                
            metrics = {
                "problem_id": pid,
                "repeat_id": rid,
                "run_mode": args.run_mode,
                "accuracy": 1 if is_correct else 0,
                "total_tokens": total_tokens,
                "draft_tokens": draft_tokens,
                "rejected_steps": rejected_steps,
                "total_gen_time_actual": total_wall_time,
                "total_gen_time_simulated": simulated_optimized_time,
                "total_safety_overhead": total_safety_time,
                "gen_speed_actual": total_tokens / total_wall_time if total_wall_time > 0 else 0,
                "gen_speed_simulated": total_tokens / simulated_optimized_time if simulated_optimized_time > 0 else 0
            }
            print(f"METRICS for Prob {pid} Repeat {rid}:", metrics)
            
            with open(f"{output_filename}.pickle", "wb") as f:
                pickle.dump(metadata_list, f)
            with open(f"{output_filename}.txt", "w") as f:
                pprint.pprint(metadata_list, stream=f)

if __name__ == "__main__":
    main()
