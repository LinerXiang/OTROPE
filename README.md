# OTROPE: Optimal Transport-based Robust Off-policy Evaluation for Large Language Models

<p align="center">
  <a href="https://openreview.net/profile?id=~Liner_Xiang1">Liner Xiang</a> ·
  <a href="https://onepounchman.github.io/">Wenbo Zhang</a> ·
  <a href="https://hengruicai.github.io/">Hengrui Cai</a>
</p>

**Accepted · Official implementation**

## 🏠 About

We present **OTROPE**, a likelihood-free framework for off-policy evaluation of
large language models. Evaluating a new target LLM with limited human feedback
from a behavior model is challenging: the two models induce different response
distributions, and token-level likelihoods are often unavailable in black-box settings.

OTROPE uses optimal transport to align labeled behavior samples with unlabeled
target samples in a semantic embedding space. It combines the target proxy mean
with OT-weighted human-labeled residuals, correcting proxy errors without behavior-policy
modeling or density-ratio estimation. Proxy predictions can be formed by a
Mixture-of-Experts (MoE) router or majority voting over smaller LLM judges.

The paper establishes consistency and convergence rates under its stated assumptions
and evaluates OTROPE on synthetic experiments, RewardBench 2, and Chatbot Arena.

## 🧭 Method

Let $\mathcal{D}_1=\{(z_i,g^*(z_i))\}_{i=1}^{n}$ denote labeled samples from the
behavior policy and $\mathcal{D}_2=\{\tilde z_j\}_{j=1}^{N}$ denote unlabeled
samples from the target policy. Here $g^*$ is the ground-truth evaluator,
$\hat g$ is a proxy evaluator, and $e(\cdot)$ maps samples into a semantic embedding space.

**1. Align the distributions.** OTROPE learns source weights by minimizing the
Wasserstein distance between the reweighted behavior samples and the target
samples in embedding space (Eq. 4):

$$
w^* \in \arg\min_{w\in\Delta_n}
W\!\left(
\sum_{i=1}^{n}w_i\delta_{e(z_i)},
\frac{1}{N}\sum_{j=1}^{N}\delta_{e(\tilde z_j)}
\right).
$$

Here $\Delta_n=\{w\in\mathbb{R}^n: w_i\geq 0,\ \sum_i w_i=1\}$ is the probability
simplex, and $\delta_{e(z)}$ is a point mass at the embedding of $z$.
The implementation uses normalized Euclidean costs, log-domain Sinkhorn iterations,
and Adam to optimize softmax-parameterized source weights.

**2. Correct the proxy estimate.** The OTROPE estimator combines the target proxy
mean with weighted residuals from the labeled behavior samples (Eq. 6):

```math
\widehat{V}_{\mathrm{OTROPE}}(\pi)
= \frac{1}{N}\sum_{j=1}^{N}\hat{g}(\tilde{z}_{j})
+ \sum_{i=1}^{n}w_{i}^{*}\bigl(g^{*}(z_{i})-\hat{g}(z_{i})\bigr).
```

The first term is the **target proxy mean**; the second is the
**OT-weighted residual correction**.

Behavior samples that better represent the target distribution receive greater
influence in the correction. This yields a doubly robust-style estimator without
requiring token-level likelihoods or explicit density ratios. In the real-data
experiments, $\hat g$ is constructed using MoE or majority voting, and the estimated
policy value is the target model's expected preference win rate.

## 🏆 Performance

OTROPE evaluates target-model win rates against an opponent pool. The paper studies
Claude 3.5 Sonnet and a mixed Claude/human target on RewardBench 2, and GPT-4 and
GPT-3.5-Turbo on Chatbot Arena.

Selected results from **Table 1 of the paper** are shown below as mean absolute
error, with standard deviation in parentheses. Lower is better.

| Evaluator | RewardBench 2: Mixed | RewardBench 2: Claude 3.5 Sonnet | Arena: GPT-4 | Arena: GPT-3.5-Turbo |
|---|---:|---:|---:|---:|
| OTROPE + MoE | 0.071 (0.046) | 0.103 (0.077) | 0.049 (0.018) | 0.010 (0.007) |
| OTROPE + majority voting | 0.066 (0.059) | 0.088 (0.083) | 0.029 (0.014) | 0.013 (0.012) |
| DeepSeek-V3.1 | 0.164 | 0.159 | 0.153 | 0.228 |

The five proxy judges are Phi-4-mini-instruct, Gemma-7B-IT, DeepSeek-LLM-7B-Chat,
Llama-3.1-8B-Instruct, and Qwen2.5-14B-Instruct. DeepSeek-V3.1 is a separate
strong-judge comparison. Additional baselines include IS, DR, PPI, and PPI++,
with DM and raw OT used for ablation studies.

## 📐 Set up

Use Python 3.10 or 3.11 and a Linux/CUDA environment compatible with vLLM for
judge inference and embedding extraction.

```bash
git clone https://github.com/LinerXiang/OTROPE.git
cd OTROPE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the commands below from the repository root. Some judge models require
Hugging Face access approval and authentication.

## 📊 Data Preparation

The pipeline constructs preference pairs, generates judge verdicts, aggregates
scores, and extracts semantic embeddings.

### 1. Construct preference pairs

```bash
# RewardBench 2: allenai/reward-bench-2
bash scripts/prepare_realdata_rb2.sh

# Chatbot Arena: lmsys/chatbot_arena_conversations
bash scripts/prepare_realdata_chatbot.sh
```

The launchers configure the target, behavior, and opponent models. They write
`real_data/reward-bench-2/reward-bench-2.json` and
`real_data/chatbot_arena_conversations/chatbot_arena_conversations.json`, respectively.
`whether_train=1` identifies behavior samples and `whether_train=0` identifies target samples.

### 2. Generate judgments

Select one dataset and keep the same `DATA` value throughout the remaining preparation steps:

```bash
export DATA=reward-bench-2
# For Arena: export DATA=chatbot_arena_conversations
GPU_IDS="0 1 2 3" bash scripts/gen_eval.sh
```

This runs the five proxy judges and merges their shards into
`real_data/${DATA}/judge_<model>.json`. Use `GPU_IDS="0"` for a single GPU.
The raw shards are retained under `real_data/${DATA}/shards/`.

### 3. Aggregate scores

```bash
bash scripts/training_data_ready.sh
```

Judgments and ground-truth labels are aligned by sample ID and saved as
`combine_train.csv` and `combine_test.csv` under `real_data/${DATA}/`.

### 4. Generate embeddings

```bash
bash scripts/get_embedding.sh
```

BGE-M3 separately embeds the candidate and opponent prompt–response pairs.
Outputs are `embed_bge-m3_train.safetensors` and `embed_bge-m3_test.safetensors`,
along with `.ids.json` sidecars for checking sample alignment. Keep each sidecar
with its embedding file.

## 🚀 Training and Evaluation

Each launcher trains the MoE router, saves source/target proxy predictions, and
calls `OTROPE.estimate_policy_value` to compute OTROPE and baseline estimates.

```bash
# RewardBench 2: Claude 3.5 Sonnet
bash scripts/run_reward-bench-2.sh

# Chatbot Arena: GPT-4
bash scripts/run_chatbot.sh
```

The default configurations use 10 replications and 50 MoE epochs:

| Parameter | RewardBench 2: Claude | Arena: GPT-4 |
|---|---:|---:|
| Labeled samples | 1,400 | 1,600 |
| MoE hidden dimensions | 64, 32 | 64, 32 |
| MoE learning rate | 0.0003 | 0.001 |
| Batch size | 256 | 256 |
| OT regularization | 0.10 | 0.05 |
| PCA dimension | 128 | 128 |
| Sinkhorn iterations | 2,000 | 2,000 |
| Outer iterations | 1,000 | 1,000 |
| OT learning rate | 0.01 | 0.01 |

Override GPU allocation, sample sizes, or repetition counts as needed:

```bash
GPU_IDS="0 1" SIZES="400 800 1200" N_REP=10 JOBS_PER_GPU=1 bash scripts/run_reward-bench-2.sh
```

| Variable | Meaning |
|---|---|
| `GPU_IDS` | Space-separated GPU indices; default `0` |
| `SIZES` | Space-separated labeled sample sizes |
| `N_REP` | Number of replications; default `10` |
| `JOBS_PER_GPU` | Concurrent jobs per GPU; default `1` |

Results, sampled IDs, predictions, and logs are saved under
`real_data/<dataset>/replication/`. To produce summary tables and plots:

```bash
python -m OTROPE.summary --results-dir real_data/reward-bench-2/replication
python -m OTROPE.summary --results-dir real_data/chatbot_arena_conversations/replication
```


## 🧪 Simulations

```bash
# Sample-size study with a fixed proxy
python simulation/simulation_size.py --device cuda:0 --n_rep 100

# Sample-size study with a fitted linear proxy
python simulation/simulation_size_train.py --device cuda:0 --n_rep 100

# Distribution-shift-strength study
python simulation/simulation_score.py --device cuda:0 --n_rep 200
```

Shift levels and predictor definitions are set in the simulation source. The
fixed-proxy sample-size script currently uses `-3x`, while Section 5.1 specifies
`-x`; the fixed and trained scripts default to shifts of 1 and 3, respectively.
Set these for the desired paper experiment. CPU execution is available with
`--device cpu`.

## 📁 Code Structure

```text
OTROPE/
├── data_io.py                # Shared data loading, ID alignment, and PCA
├── otrope_weights.py         # Sinkhorn and OT weight optimization
├── estimators.py             # OTROPE and baseline estimators
├── estimate_policy_value.py  # Evaluation entry point
└── summary.py                # Result tables and plots
data_preparation/            # Benchmark preparation, judging, embeddings, and MoE
simulation/                  # Controlled experiments
scripts/                     # Bash launchers
```

## 📄 Citation

The full citation will be added with the public paper link and author information.
