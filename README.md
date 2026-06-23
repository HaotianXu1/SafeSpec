<div align="center">

# 🛡️ SafeSpec

### **Unlocking Secure Inference Acceleration via Dynamic Reflective Sampling**

### 🎉 Accepted to ICML 2026 🎉

[![Conference](https://img.shields.io/badge/ICML-2026-8A2BE2.svg)](https://icml.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.19755-b31b1b.svg)](https://arxiv.org/abs/2606.19755)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Safety%20Heads-yellow.svg)](https://huggingface.co/HaotianXu1/safespec-safety-heads)

</div>

<p align="center">
  <img src="assets/overview.png" width="940"><br>
  <em>Overview of SafeSpec: a dual-head verification mechanism inside the target model jointly assesses
  semantic <b>quality</b> and <b>safety</b> in a single forward pass, with rollback + reflective multi-sampling to recover safe continuations.</em>
</p>

---

## 📖 Abstract

Speculative inference accelerates large language model (LLM) decoding but provides no inherent safety guarantees. Existing safety defenses are largely incompatible with speculative inference: they either introduce additional computation or disrupt the draft–verify mechanism, negating acceleration benefits. This reveals a fundamental incompatibility between current safety methods and speculative decoding.

We propose **SafeSpec**, a safety-aware speculative inference framework that integrates risk estimation directly into the verification process. SafeSpec attaches a **lightweight latent safety head** to the target model to jointly evaluate semantic validity and safety in a **single forward pass**. When unsafe generations are detected, SafeSpec applies **rollback and safety-guided reflective multi-sampling** to recover safe continuations rather than terminating generation.

We model jailbreak attacks as distributional shifts over generative trajectories, where adversarial prompts increase the probability of harmful continuations without eliminating safe ones. Under this model, SafeSpec performs risk-aware trajectory recovery within the speculative decoding process, achieving a substantially improved safety–efficiency trade-off and demonstrating that speculative acceleration and inference-time safety can be jointly optimized.

📄 **Paper:** [arXiv:2606.19755](https://arxiv.org/abs/2606.19755)

## ⚙️ Method

At each step the draft model proposes a candidate segment; the target model then **verifies it with two heads at once** — an **LM head** for quality scoring and a **latent safety head** reading the last hidden state. Based on the verdict, SafeSpec runs in one of two modes:

- **General Mode** — accept / regenerate the segment by quality, as in standard speculative reasoning.
- **Safety Mode** — when the safety head flags risk, **roll back**, insert a reflection prompt, and **reflectively multi-sample** a safe continuation.

Three run modes are provided (`spec_reason.py` / `spec_reason_ppl.py`):

| Mode | Description |
|------|-------------|
| `target_only` | target model alone (accuracy ceiling) |
| `speculative` | draft proposes, target scores/accepts |
| `spec_ppl` | speculative **+ safety head + rollback/recovery** — the SafeSpec defense |

Supported model pairs: **Qwen3-32B / Qwen3-1.7B** and **DeepSeek-R1-Distill-Llama-70B / -8B**.

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/HaotianXu1/SafeSpec.git
cd SafeSpec
pip install -r requirements.txt
```

### Quick start

```bash
# 1) launch vLLM servers (base + draft + safety-head pooling instance)
bash specreason/run_deepseek_servers.sh
bash specreason/run_safety_pool.sh

# 2) run reasoning — pick a mode via --run_mode {target_only|speculative|spec_ppl}
python specreason/spec_reason_ppl.py --help

# 3) (optional) reproduce benchmarks
bash specreason/run_math_gsm8k_official.sh
```

> ⚙️ The launch scripts contain machine-specific paths (model dirs, GPU ids) — edit them for your environment. The framework talks to vLLM via an OpenAI-compatible client (`api_key="EMPTY"`, localhost); no external API key required.

### Training a safety head

The full pipeline lives in `safety_head/`: generate draft rollouts (`generate.py`) → safety-label (`label.py`) → extract target hidden states (`extract_features.py`) → assemble the training set (`build_mix_*.py`) → train the MLP head (`train_head_cached.py`).

## 🤗 Pretrained Safety Heads

Weights are on the Hugging Face Hub (GitHub is not for large binaries):

> **[`HaotianXu1/safespec-safety-heads`](https://huggingface.co/HaotianXu1/safespec-safety-heads)** *(update with your actual HF repo)*

| Head | Target model | hidden dim | size |
|------|-------------|-----------|------|
| `qwen3-32b` | Qwen3-32B | 5120 | 51 MB |
| `deepseek-r1-70b` | DeepSeek-R1-Distill-Llama-70B | 8192 | 129 MB |

Each is a small MLP over the target model's last-layer mean-pooled hidden states.

## 📁 Repository structure

```
SafeSpec/
├── specreason/                       # speculative reasoning framework
│   ├── spec_reason.py                #   target_only / speculative modes
│   ├── spec_reason_ppl.py            #   spec_ppl mode (safety head + recovery)
│   └── run_*.sh                      #   vLLM server / benchmark launch scripts
└── safety_head/                      # safety head training pipeline
    ├── generate.py · label.py · extract_features.py · train_head_cached.py
    ├── safety_head.py                #   safety head model + pooling
    └── build_mix_*.py · *.sh
```

## 📝 Citation

```bibtex
@article{xu2026safespec,
  title   = {SafeSpec: Unlocking Secure Inference Acceleration via Dynamic Reflective Sampling},
  author  = {Xu, Haotian and Zhang, Zeyang and Li, Linbao and Zheng, Huadi and Li, Yu and Zhuo, Cheng},
  journal = {arXiv preprint arXiv:2606.19755},
  year    = {2026}
}
```

## 🙏 Acknowledgements

SafeSpec is built on top of **[SpecReason](https://github.com/ruipeterpan/specreason)** (Pan et al., NeurIPS 2025), whose step-level speculative reasoning framework we extend with safety-aware verification. We thank the authors for open-sourcing their code.

## 📄 License

Released under the MIT License *(add a LICENSE file)*.
