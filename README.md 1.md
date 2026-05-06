# Multi-Objective Lightweight Specialized Language Models

> Compressing BERT for real-world deployment using Pruning, Quantization, and Knowledge Distillation — with trade-off analysis via Pareto Frontier.

**Internship Project | GITAM University | Batch No: 16**  
**Team:** T. Kiran Kumar (2023002462) · K. Chandhan Sai (2023004196) · G. Sai Thrishul (2023002221) · Vasu Reddy K S (2023002549)

---

## Overview

Large Language Models like BERT and GPT deliver excellent NLP performance but are too large and computationally expensive for mobile and real-time applications. This project applies **model compression techniques** to reduce model size and improve inference speed while maintaining acceptable accuracy.

The project focuses on three key compression techniques:
- **Pruning** — Removing unimportant weights from the model
- **Quantization (INT8)** — Reducing weight precision from 32-bit float to 8-bit integer
- **Knowledge Distillation** *(discussed as future scope)*

Performance is evaluated across three metrics — **Accuracy**, **Latency**, and **Model Size** — and trade-offs are visualized using a **Pareto Frontier**.

---

## Results Summary

| Metric | Baseline Model | Quantized Model |
|---|---|---|
| Accuracy | 0.72 | 0.68 |
| Latency (sec/sample) | 0.012 | 0.006 |
| Model Size | 16.8 MB | 8.5 MB |
| Speed Improvement | 1x (base) | **2x faster** |
| Size Reduction | 1x (base) | **2x smaller** |
| Accuracy Drop | — | 4% lower |

> The Quantized model achieves **2x speed** and **2x size reduction** with only **4% accuracy loss** — a strong trade-off for real-world edge deployment.

---

## Dataset

- **Name:** [AG News](https://huggingface.co/datasets/ag_news)
- **Categories:** World, Sports, Business, Science/Technology
- **Training samples used:** 1,500
- **Test samples used:** 300

---

## Model

- **Base Model:** [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) (4.4M parameters)
- **Architecture:** 2 Transformer layers, 128 hidden dimensions, 2 attention heads
- **Loaded via:** HuggingFace Transformers library
- **Evaluation Model:** [`textattack/distilbert-base-uncased-ag-news`](https://huggingface.co/textattack/distilbert-base-uncased-ag-news)

---

## Techniques Used

### Pruning
- Method: **L1 Unstructured Pruning** (`torch.nn.utils.prune`)
- Amount: **30%** of smallest weights removed from the classifier layer

### Quantization
- Method: **Dynamic INT8 Quantization** (`torch.quantization.quantize_dynamic`)
- Applied to: Linear layers
- No retraining required

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Tokenizer max length | 128 tokens |
| Training epochs | 2 |
| Learning rate | 3e-5 |
| Batch size | 8 |
| Optimizer | AdamW |
| Hardware | CPU only (edge device simulation) |

---

## Project Structure

```
├── Internship.ipynb       # Main notebook with full pipeline
├── finalint.pptx          # Project presentation slides
└── README.md
```

---

## Installation & Usage

```bash
pip install torch transformers datasets evaluate matplotlib
```

Then open and run `Internship.ipynb` in Jupyter or Google Colab.

---

## Future Work

1. **Transfer Learning** — Fine-tune larger pre-trained models (ResNet50, VGG16, EfficientNet) to push accuracy beyond 99%
2. **Higher Input Resolution** — Use 224×224 or 256×256 for richer feature capture
3. **Knowledge Distillation** — Train a small student model from a large teacher model
4. **Multi-modal Input** — Combine T1, T2, FLAIR MRI modalities for richer diagnostic input
5. **Larger Datasets** — Train on the full AG News dataset (120,000 samples) for higher accuracy

---

## References

1. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL-HLT.
2. Sanh et al. (2019). *DistilBERT, a distilled version of BERT*. arXiv:1910.01108.
3. Jiao et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding*. EMNLP.
4. Han et al. (2015). *Learning both Weights and Connections for Efficient Neural Networks*. NeurIPS.
5. Jacob et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR.
6. Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531.
7. Zhang et al. (2015). *Character-level Convolutional Networks for Text Classification*. NeurIPS. (AG News Dataset)
8. [HuggingFace Transformers](https://huggingface.co/transformers)
9. [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)

---

## Topics

`nlp` `bert` `model-compression` `quantization` `pruning` `knowledge-distillation` `edge-ai` `pytorch` `huggingface` `text-classification` `pareto-frontier` `lightweight-models`
