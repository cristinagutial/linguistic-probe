# 🔬 Cross-lingual NPI Probe

**Does mBERT "know" when a Negative Polarity Item is licensed and does it know it equally in English and Spanish?**

This project probes `bert-base-multilingual-cased` (mBERT) on **Negative Polarity Item (NPI) licensing** a linguistically rich phenomenon that requires sensitivity to long-distance syntactic and semantic relationships, not just local patterns.

This connects to a core question in current NLP research: *what do language models actually learn about grammar, and does it generalize cross-lingually?*

---

## What is an NPI?

NPIs are expressions that are grammatical **only** in certain *licensing contexts* canonically negation, but also polar questions and conditionals:

| Licensed ✓ | Unlicensed ✗ |
|---|---|
| She doesn't have **any** money. | ~~She does have **any** money.~~ |
| Has she **ever** been to Paris? | ~~She has **ever** been to Paris.~~ |
| No ha llamado **nadie**. | ~~Ha llamado **nadie**.~~ |
| ¿Ha llamado **nadie**? | — |

Crucially, the licensor can appear far from the NPI (*"The teacher didn't think that **any** student had cheated"*) which means probing NPIs tests **long-distance syntactic/semantic sensitivity**.

---

## NPI Types Tested

| Language | Weak NPIs | Strong NPIs |
|---|---|---|
| English | *any*, *ever*, *yet* | *at all* |
| Spanish | *nadie*, *nunca*, *ningún* | *jamás*, *en absoluto* |

**Weak NPIs** are licensed in all downward-entailing contexts.  
**Strong NPIs** require strict negation: they are more restricted and thus harder to model.

---

## Licensing Contexts

| Licensor | Example (EN) | Example (ES) |
|---|---|---|
| `negation` | *She **doesn't** have any money.* | *No ha llamado nadie.* |
| `question` | *Did you find **any** mistakes?* | *¿Ha llamado nadie?* |
| `conditional` | ***If** you have any questions, ask.* | ***Si** nunca tienes dudas, pregunta.* |
| `negation_long_distance` | *She **didn't** think that any student cheated.* | *No creyó que nadie hubiera copiado.* |

---

## Method

Each minimal pair holds the NPI word constant and varies only the licensing context. We place `[MASK]` at the NPI position and compare:

```
Δ log p = log p(NPI | licensed context) − log p(NPI | unlicensed context)
```

- **Δ > 0** → model assigns higher probability to the NPI in the grammatical context ✓  
- **Δ < 0** → model prefers the NPI in the ungrammatical context ✗  

Multi-token NPIs (*at all*, *en absoluto*) are scored via **iterative masked scoring** — each sub-token is scored while conditioning on previously predicted sub-tokens.

---

## Results

> Run the probe yourself to generate results! See **Getting Started** below.

Saved to `results/`:
- `results_raw.csv` — per-pair scores, log-probs, model decisions
- `results_accuracy.csv` — accuracy by language, by licensor, and by language × licensor
- Plots from the analysis notebook

---

## Project Structure

```
linguistic-probe/
│
├── data/
│   └── sentences.json          # 16 NPI minimal pairs (EN + ES)
│
├── notebooks/
│   └── analysis.ipynb          # Visualizations & linguistic discussion
│
├── results/                    # Auto-generated after running the probe
│   ├── results_raw.csv
│   ├── results_accuracy.csv
│   └── *.png
│
├── linguistic_probe.py         # Main probing script
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/linguistic-probe.git
cd linguistic-probe

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the probe (downloads mBERT on first run, ~700MB)
python linguistic_probe.py

# 4. Filter by language or licensor type
python linguistic_probe.py --language es
python linguistic_probe.py --licensor negation_long_distance
python linguistic_probe.py --language en --licensor question

# 5. Explore results in the notebook
jupyter notebook notebooks/analysis.ipynb
```

---

## Extending This Project

- **Add more languages** — French (*personne*, *jamais*), Italian (*nessuno*, *mai*), German (*niemand*, *jemals*) for a fuller cross-lingual picture
- **Add more NPI types** — idioms like *lift a finger*, *sleep a wink*; comparative NPIs like *any more*
- **Compare models** — run the same probe on monolingual BERT (`dccuchile/bert-base-spanish-wwm-cased` for ES) to test whether multilingual pretraining helps or hurts NPI sensitivity
- **Surprisal-based scoring** — instead of masked token probability, use full-sentence log-probability with an autoregressive model (e.g. GPT-2)
- **Polarity scale** — test scalar NPIs (*a single*, *a red cent*) which require even stricter licensing than *any*

---

## Background Reading

- Ladusaw (1980) — *Polarity Sensitivity as Inherent Scope Relations* (the foundational NPI theory)  
- Giannakidou (1998) — *Polarity Sensitivity as (Non)Veridical Dependency*  
- Warstadt et al. (2019) — [BLiMP: The Benchmark of Linguistic Minimal Pairs for English](https://arxiv.org/abs/1912.00582)  
- Marvin & Linzen (2018) — [Targeted Syntactic Evaluation of Language Models](https://arxiv.org/abs/1808.09031)  
- Goldberg (2019) — [Assessing BERT's Syntactic Abilities](https://arxiv.org/abs/1901.05287)  
- Jumelet & Hupkes (2018) — [Do Language Models Understand Anything? On the Ability of LSTMs to Understand Negative Polarity Items](https://arxiv.org/abs/1811.00225)

---

## Author

M.Sc. Computational Linguistics student @ University of Stuttgart (IMS)  
Background in Linguistics · Interested in speech processing, NLP & LLMs
