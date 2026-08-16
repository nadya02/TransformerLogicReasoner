# TransformerLogicReasoner

A decoder-only transformer, trained from scratch, that learns to continue and generate formal logical proofs over biomedical ontologies expressed in Description Logic (OWL EL).

## Why this exists

Automated reasoning over ontologies is normally handled by symbolic reasoners (e.g. ELK) that apply fixed inference rules to derive proofs. This project asks a different question: **can a small transformer model learn the structure of these proofs directly from data, without being given the inference rules?**

Most prior work testing LLM reasoning ability uses large pretrained models in zero/few-shot settings on math or first-order logic. This project instead trains a small model (~1M parameters) *from scratch*, specifically on Description Logic proofs — an area with little prior work — and tests two distinct capabilities: continuing a partial proof, and generating a full proof from just the premises.

## How it works

- **Architecture**: decoder-only transformer (GPT-style), implemented from scratch — 8 transformer blocks, 4 attention heads per block, 128-dimensional embeddings, 256-character context window, byte-level tokenization.
- **Dataset**: generated from 150 biomedical ontologies pulled from [BioPortal](https://bioportal.bioontology.org/), covering domains from anatomy to gene function. Proofs were produced using the [ELK reasoner](https://code.google.com/archive/p/elk-reasoner/) via the Evee library, which decomposes entailments into their inference steps. ~1M raw proof examples were generated; after deduplication and filtering by proof size, ~276K examples remained for training.
- **Training**: Adam optimizer with a 10K-step learning rate warmup to 1e-4, batch size 32, ~600K batches, trained on a single RTX 2080Ti (~12 hours).

## Results

| Task | NLL Loss | Compression Loss (bits/byte) |
|---|---|---|
| Proof continuation | 0.36 | 0.54 |
| Proof generation from premises | 5.43 | 7.57 |

The model performs strongly on **proof continuation** — given a partial proof, it reliably predicts how it should continue, and generalizes to ontologies it wasn't trained on. It performs poorly on **proof generation from premises alone** — the harder task of constructing a full proof from scratch.

This gap is the most interesting finding of the project: skill learned on one task (pattern-continuation) does not transfer to a structurally different task (derivation from first principles), even though both involve the "same" proofs. The full writeup discusses why, and proposes concrete next steps (training jointly on premises + proofs, fine-tuning the continuation model, or scaling to a larger pretrained backbone).

## Tech stack

`Python`, `PyTorch`[, add any others — e.g. NumPy, specific tokenizer libs]

## Repo structure

```
[fill in — e.g.]
├── data/           # dataset generation scripts (BioPortal → ELK → JSON proofs)
├── model/          # transformer architecture
├── train.py        # training loop
├── evaluate.py     # NLL / compression loss evaluation
└── ...
```

## Background

This project was completed as a bachelor's thesis at Vrije Universiteit Amsterdam (2024).
