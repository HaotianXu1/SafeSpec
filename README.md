<div align="center">

# 🛡️ SafeSpec

**Safety-aware speculative reasoning for large reasoning models**

*A lightweight safety head + speculative reasoning framework that screens chain-of-thought generation for harmful content — without hurting benign accuracy.*

</div>

---

## ✨ Overview

Large reasoning models (DeepSeek-R1, Qwen3, …) generate long chains of thought that current safety filters — built for single-turn answers — fail to monitor. **SafeSpec** trains a tiny **safety head** on a base model's hidden states and plugs it into a **speculative reasoning** loop, screening every reasoning step and recovering before the model complies with a jailbreak.

Supported model pairs:

| Role | Qwen | DeepSeek |
|------|------|----------|
| **Base** (scorer) | Qwen3-32B | DeepSeek-R1-Distill-Llama-70B |
| **Draft** (proposer) | Qwen3-1.7B | DeepSeek-R1-Distill-Llama-8B |

## ⚙️ How it works

Each reasoning step goes through:

```
 ┌─────────┐  step   ┌──────────────┐  quality   ┌─────────────┐
 │  draft  │ ──────▶ │ base scorer  │ ─ score ─▶ │  accept?    │
 │  model  │         │   (0–9)      │            │ (≥ thresh)  │
 └─────────┘         └──────────────┘            └─────────────┘
                            │
                            ▼  hidden states
                     ┌──────────────┐  unsafe   ┌──────────────────────┐
                     │ safety head  │ ────────▶ │ rollback + recovery   │
                     │ (MLP, 0–1)   │           │ (insert warning,      │
                     └──────────────┘           │  regenerate / refuse) │
                                                └──────────────────────┘
```

Three run modes (in `spec_reason.py` / `spec_reason_ppl.py`):

- **`target_only`** — base model alone (accuracy ceiling).
- **`speculative`** — draft proposes, base scores/accepts (SpecReason).
- **`spec_ppl`** — speculative **+ safety head + rollback/recovery** (the SafeSpec defense).

## 📁 Repository structure

```
SafeSpec/
├── requirements.txt
├── specreason/                       # speculative reasoning framework
│   ├── spec_reason.py                #   target_only / speculative modes
│   ├── spec_reason_ppl.py            #   spec_ppl mode (safety head + recovery)
│   ├── run_deepseek_servers.sh       #   launch base + draft vLLM servers
│   ├── run_safety_pool.sh            #   launch safety-head pooling instance
│   ├── run_bench_ds_n100.sh          #   benchmark driver (3 modes × datasets)
│   ├── run_math_gsm8k_official.sh    #   official-config benchmark (GSM8K/MATH)
│   └── run_spec_official_parallel.sh #   sharded parallel runner
└── safety_head/                      # safety head training pipeline
    ├── generate.py                   #   draft model rollouts
    ├── label.py                      #   safety labeling (guard model)
    ├── extract_features.py           #   base-model hidden-state extraction
    ├── train_head_cached.py          #   train the MLP safety head
    ├── safety_head.py                #   safety head model + pooling
    ├── build_mix_*.py                #   training-set assembly
    └── *.sh                          #   data-gen / extract / training scripts
```

## 🚀 Installation

```bash
git clone https://github.com/HaotianXu1/SafeSpec.git
cd SafeSpec
pip install -r requirements.txt
```

## 🤗 Pretrained safety heads

Weights are hosted on the Hugging Face Hub (GitHub is not for large binaries):

> **`HaotianXu1/safespec-safety-heads`**  *(update with your actual HF repo)*

| Head | Base model | hidden dim | size |
|------|-----------|-----------|------|
| `qwen3-32b` | Qwen3-32B | 5120 | 51 MB |
| `deepseek-r1-70b` | DeepSeek-R1-Distill-Llama-70B | 8192 | 129 MB |

Each is a small MLP over the base model's last-layer mean-pooled hidden states (threshold 0.5–0.6).

## 🏃 Quick start

```bash
# 1) launch vLLM servers (base + draft + safety-head pooling instance)
bash specreason/run_deepseek_servers.sh
bash specreason/run_safety_pool.sh

# 2) run reasoning — pick a mode via --run_mode {target_only|speculative|spec_ppl}
#    spec_ppl = the SafeSpec defense (safety head + rollback/recovery)
python specreason/spec_reason_ppl.py --help

# 3) (optional) reproduce benchmarks
bash specreason/run_math_gsm8k_official.sh
```

> ⚙️ **Configuration:** the launch scripts contain machine-specific paths (model dirs, GPU ids). Edit them to match your environment before running. The framework talks to the vLLM servers via an OpenAI-compatible client (`api_key="EMPTY"`, localhost) — no external API key required.

## 🧠 Training a safety head

```bash
# generate rollouts → label → extract base hidden states → train MLP head
# see safety_head/*.sh and *.py for the full pipeline
```

## 📊 Datasets

Benchmark datasets (GPQA-Diamond, GSM8K, MATH, XSTest, HellaSwag) are **not** bundled — fetch them from their official sources / the Hugging Face Hub.

## 🙏 Acknowledgements

The optional SafeDecoding baseline is **not** included; it requires the upstream [SafeDecoding](https://github.com/uw-nsl/SafeDecoding) repo.

## 📄 License

Released under the MIT License *(add a LICENSE file)*.
