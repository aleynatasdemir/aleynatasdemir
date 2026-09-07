<h1 align="center">Habibe Aleyna Taşdemir</h1>

<p align="center">
  <b>ML Engineer · NLP &amp; Information Retrieval</b><br>
  Turkish foundation models, dense &amp; late-interaction retrieval, embedding training
</p>

<p align="center">
  <a href="https://huggingface.co/aleynatasdemir"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?style=flat-square&logoColor=black" alt="Hugging Face"></a>
  <a href="https://arxiv.org/search/?searchtype=author&query=Tasdemir%2C+H+A"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://www.linkedin.com/in/aleyna-tasdemir/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="mailto:aleynattasdemir@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>
</p>

---

I build language models for Turkish — from tokenizer design and pretraining through retrieval evaluation. Most of my work sits at the intersection of **encoder pretraining** and **information retrieval**: what a model has to learn during pretraining so that it retrieves well afterwards.

Two of these efforts are now on arXiv, with open weights on Hugging Face.

- Computer Engineering, Ankara University
- Dense retrieval for e-commerce search at scale
- Currently: extending the Mogan line toward a Turkish decoder trained from scratch

---

## Research

**MoganBERT-TR: A Turkish Encoder Foundation Model Trained from Scratch with a CLM-to-MLM Curriculum**
[`arXiv:2608.25768`](https://arxiv.org/abs/2608.25768)

A Turkish encoder trained from zero — no multilingual checkpoint, no continued pretraining. The central result is a two-phase curriculum: a causal-LM warmup followed by masked-LM training. At a 25% CLM / 75% MLM mixture, this yields a **2.7–6.9× improvement on retrieval benchmarks** over a pure-MLM baseline of the same size and token budget.

**MoganColBERT-TR: A Late-Interaction Multi-Vector Retrieval Model for Turkish**
[`arXiv:2608.26344`](https://arxiv.org/abs/2608.26344)

A late-interaction (ColBERT-style) retriever built on top of the Mogan encoder line, trained on retrieval-only pairs with passage-level handling for long documents. Evaluated on TurkColBERT.

---

## MoganAI — a Turkish foundation model family, built from scratch

A team effort with Furkan Yılmaz and Faruk Gözay, self-funded, run on rented GPUs.

| | |
|---|---|
| **Corpus** | ~10 TB of raw Turkish text, cleaned and deduplicated down to a **200B-token** pretraining corpus |
| **Tokenizer** | Custom SentencePiece Unigram, 50,048 vocab, identity normalization, indentation-preserving for code |
| **Architecture** | ModernBERT-style encoder, ~149M parameters |
| **Curriculum** | Two-phase **CLM → MLM**, validated by ablations at a 10B-token budget |
| **Hardware** | 4×H100, self-funded |
| **Derived models** | Dense embedding model (MTEB-TR) + ColBERT late-interaction retriever (TurkColBERT) |

**Engineering notes from the run** — the parts that actually cost time:

- Raised MFU from **9% → ~21.5%** with fused cross-entropy, varlen sequence packing and `torch.compile`
- Fixed a causality leak traced to a stray `.transpose` in varlen attention
- Eliminated CUDA OOM from materialized fp32 logits by fusing the loss
- Caught a varlen sliding-window bug (effective window of 65 instead of 129 tokens) that was quietly degrading retrieval
- Built an evaluation harness: domain-stratified BPC monitoring, UD Turkish POS linear probes, MLM probe sets across legal / academic / code / general domains

---

## Applied work

**Dense retrieval for e-commerce search**
First-stage candidate retrieval over large-scale product and query data — embedding model fine-tuning, hard negative mining from behavioral signals, ANN indexing, and retrieval quality analysis. Findings along the way: a train/serve prefix mismatch in multilingual GTE fine-tuning (bare-query serving beat prefixed serving), and label-scale saturation in weighted MultipleNegativesRankingLoss.

**AI Triage — Turkish emergency-department triage LLM**
LoRA fine-tuning on a 26B instruction model for Turkish ED triage practice, trained on 6,475 synthetic dialogues, served behind a RAG-backed clinical decision layer (FastAPI · PostgreSQL · vector DB · React).

**FinansAI / QuantTrade**
End-to-end algorithmic trading pipeline on BIST data — feature engineering, CatBoost/LightGBM forecasting, T+1 portfolio simulation, plus a financial LLM fine-tuned on 161K+ news articles producing structured analysis for the ML pipeline.

**SkinCore**
Hybrid semantic search over 15,000+ scraped cosmetic products (multimodal embeddings on MongoDB), shipped as an end-to-end iOS app with OCR-based product analysis.

---

## Open models &amp; datasets

**Mogan line** — Turkish encoder family, trained from scratch, weights open on [🤗 moganai](https://huggingface.co/moganai)

| Model | What it is |
|---|---|
| [`moganai/MoganBERT-TR`](https://huggingface.co/moganai/MoganBERT-TR) | Base encoder, ~149M params, CLM→MLM curriculum — the backbone for everything below |
| [`moganai/MoganEmbed-TR`](https://huggingface.co/moganai/MoganEmbed-TR) | Dense single-vector embedding model, contrastive + distillation trained, evaluated on MTEB-TR |
| [`moganai/MoganColBERT-TR`](https://huggingface.co/moganai/MoganColBERT-TR) | Late-interaction multi-vector retriever, evaluated on TurkColBERT |
| [`moganai/turkish-resmi-gazete`](https://huggingface.co/datasets/moganai/turkish-resmi-gazete) | Turkish Official Gazette corpus |

**Earlier work**

| Model | What it is |
|---|---|
| [`BIST-Financial-Qwen-7B`](https://huggingface.co/YOUR_HF_USERNAME/BIST-Financial-Qwen-7B) | Financial LLM fine-tuned for Turkish market analysis |
| [`gemma4-26B-A4B-triage-turkish`](https://huggingface.co/YOUR_HF_USERNAME/gemma4-26B-A4B-triage-turkish) | Turkish emergency triage model |

---

## Stack

**Training** · PyTorch · Transformers · Accelerate · PEFT · TRL · DeepSpeed · Flash Attention · `torch.compile`
**Retrieval** · sentence-transformers · PyLate · FAISS · BM25 · MTEB
**Serving &amp; data** · vLLM · FastAPI · PostgreSQL · pandas · NumPy · SQL
**Classical ML** · scikit-learn · LightGBM · CatBoost

---

## Currently

- Training a Turkish decoder from scratch on an expanded corpus
- Hybrid sparse + dense retrieval (BM25 fusion) for the Mogan retrieval line
- Turkish encoder evaluation methodology — benchmark design, seed variance, hyperparameter grids

---

<p align="center">
  <i>Open to conversations about Turkish NLP, retrieval, and pretraining at small scale.</i>
</p>
