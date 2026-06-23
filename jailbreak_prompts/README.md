# Jailbreak Evaluation Prompts

The **7 jailbreak attacks** used to evaluate SafeSpec in the paper:

| File | Attack |
|------|--------|
| `AttentionShift.jsonl` | AttentionShift |
| `CodeChameleon.jsonl` | CodeChameleon |
| `DeepInception.jsonl` | DeepInception |
| `ABJ.jsonl` | ABJ |
| `ReNeLLM.jsonl` | ReNeLLM |
| `Mousetrap.jsonl` | Mousetrap |
| `H-CoT.jsonl` | H-CoT |

Each line is one attack with two fields:

- **`query`** — the underlying harmful intent (AdvBench-style benchmark item).
- **`jailbreak_prompt`** — the attack-transformed prompt actually fed to the model.

> ⚠️ **For reproducing the defensive safety evaluation only.** Model responses and judge
> verdicts have been intentionally removed — these files contain *attack inputs*, not harmful
> model outputs. The attack methods are all from prior published work.
