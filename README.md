# Spectral Fusion and VLM-Refined Captioning via Dual Attention

**A Frequency-Domain Dual Attention and Spectral Fusion Framework for Robust Image Captioning with Instruction-Tuned VLM Caption Refinement**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-MSCOCO--2014-green)](https://cocodataset.org)



---

## Overview

This repository presents a **novel image captioning architecture** that integrates **frequency-domain spectral attention**, **dual-branch visual reasoning**, and **instruction-tuned visual language model (VLM) refinement** for generating accurate, detailed, and grounded image captions.

The framework fuses **YOLOv8-based object detection** and **Xception-based scene classification** using **Discrete Cosine Transform (DCT)** in the frequency domain. It then applies a **Spectral Attention Module** using FFT to capture long-range dependencies efficiently, followed by a **GRU-based decoder** and an **LLM-guided refinement stage** for factual correctness.

---

## Novelty

* **Dual-Attention Fusion with Spectral Guidance**
Independent Bahdanau attention heads process YOLOv4 object features and Xception scene features separately before fusion — preventing the feature interference that degrades discriminability in conventional early-fusion approaches.

* **DCT-Based Frequency Fusion**
Attended feature maps from both branches are converted to the frequency domain via 2D Discrete Cosine Transform. Only low-frequency coefficients are retained, capturing dominant spatial structure while suppressing noise. The two branches are merged element-wise and reconstructed via IDCT, producing a compact joint representation.

* **FFT-Based Spectral Attention**
A 1D Fast Fourier Transform replaces standard spatial self-attention. Learnable complex spectral weights model global long-range dependencies at O(n log n) complexity — compared to O(n²) in conventional transformer self-attention.

* **Multimodal LLM Caption Refinement**
An instruction-tuned Vision-Language Model (LLaVA) acts as a post-generation editor. Using detected object lists, image embeddings, and the GRU draft caption as input, it corrects hallucinations, enriches relational descriptions, and enforces visual grounding — with no additional model retraining required.

---


## Key Features & Architecture
This framework improves upon classical CNN+RNN architectures, resulting in improved performance on the MSCOCO-2014 dataset.

### Key Contributions
* **Dual-Stream Encoder:** Object-level features from YOLOv8 + scene-level features from Xception.
* **Dual Bahdanau Attention:** Independent attention over object and scene embeddings.
* **Frequency-Domain Fusion (DCT):** Low-frequency spectral merging for compact and meaningful representations.
* **Spectral Attention Module (FFT-based):** Captures global feature interactions with reduced complexity.
* **GRU Caption Decoder:** Generates draft captions.
* **VLM Refinement:** Multimodal LLM (e.g., MiniGPT-4/LLaVA) improves caption factuality and grounding.

flowchart TD

    A[Input Image]

    %% ================= Encoders =================
    subgraph Encoders
        B[YOLOv4 Encoder<br/>(Object Detection)<br/>Fd ∈ R^(Nd × Dd)]
        C[Xception Encoder<br/>(Scene Classification)<br/>Fc ∈ R^(Nc × Dc)]
    end

    %% ================= Attention =================
    subgraph Attention Mechanism
        D[Bahdanau Attention<br/>(Object Branch)]
        E[Bahdanau Attention<br/>(Scene Branch)]
        F[Context Cd]
        G[Context Cc]
    end

    %% ================= Fusion =================
    subgraph Frequency Fusion
        H[DCT-Based Fusion<br/>2D DCT → Low-Freq Preserve<br/>Element-wise Multiply → IDCT]
        I[Ffused (Spatial Domain)]
        J[FFT-Based Spectral Attention<br/>1D FFT → Learnable Wspec<br/>Element-wise Multiply → IFFT<br/>O(N log N)]
    end

    %% ================= Captioning =================
    subgraph Caption Generation
        K[GRU Caption Decoder<br/>Draft Caption]
        L[LLaVA Refinement<br/>Inputs:<br/>• Draft Caption<br/>• YOLO Detections<br/>• Image Embeddings]
        M[Final Refined Caption]
    end

    %% Flow
    A --> B
    A --> C

    B --> D --> F --> H
    C --> E --> G --> H

    H --> I --> J --> K --> L --> M
---

## Results

Incremental evaluation on the MSCOCO-2014 Karpathy test split. Each row adds one component on top of the previous.

| Configuration | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---|---|---|---|
| Baseline (Xception + GRU) | 0.066 | 0.142 | 0.324 | 0.248 | 0.081 |
| + YOLO Object Features | 0.071 | 0.148 | 0.329 | 0.265 | 0.085 |
| + Dual Attention | 0.079 | 0.153 | 0.338 | 0.290 | 0.090 |
| + DCT Frequency Fusion | 0.084 | 0.158 | 0.345 | 0.312 | 0.094 |
| + Spectral Attention | 0.092 | 0.162 | 0.353 | 0.335 | 0.099 |
| + LLM Refinement (Full Model) | **0.105** | **0.170** | **0.372** | **0.368** | **0.108** |

The full model achieves a **59.1% improvement in BLEU-4** and **48.4% in CIDEr** over the baseline. Absolute values are lower than full-dataset benchmarks due to the 20k training subset, but relative trends are consistent.

---

## Codebase Structure

```
Spectral-Fusion-VLM-Refined-Captioning/
├── src/
│   └── improv_imgcap.py      # main runnable script (full pipeline)
├── colab nb/
│   └── *.ipynb               # experimental and ablation notebooks
├── configs/
│   └── config.yaml           # dataset paths and hyperparameters
├── scripts/
│   ├── extract_features.sh
│   ├── train_model.sh
│   ├── evaluate_model.sh
│   └── refine_captions.sh
└── README.md
```

---

## Installation

```bash
git clone https://github.com/kapish19/Spectral-Fusion-VLM-Refined-Captioning.git
cd Spectral-Fusion-VLM-Refined-Captioning

python -m venv venv
source venv/bin/activate

```

---

## Dataset Setup

Download MS COCO 2014 from [cocodataset.org](https://cocodataset.org) and organize as:

```
data/
├── train2014/
├── val2014/
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
```

Update `configs/config.yaml` with the path to your `data/` directory.

---

## Running the Project

**Feature extraction**
```bash
bash scripts/extract_features.sh
```

**Training**
```bash
bash scripts/train_model.sh
```

**Evaluation**
```bash
bash scripts/evaluate_model.sh
```

**Caption refinement** (VLM post-processing, applied at inference only)
```bash
bash scripts/refine_captions.sh
```

The full pipeline can also be run directly:
```bash
python src/improv_imgcap.py
```

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Feature projection dim | 256 |
| GRU hidden size | 512 |
| DCT low-freq keep ratio | 0.25 |
| Beam search width | 5 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Training subset | 20,000 images |

---

## Base Repository

This work extends the baseline architecture from [abdelhadie-almalla/image_captioning](https://github.com/abdelhadie-almalla/image_captioning), enhancing the original CNN-GRU pipeline with dual attention, frequency-domain fusion, spectral attention, and instruction-tuned VLM refinement.

---

*Netaji Subhas University of Technology, Delhi*

**A Frequency-Domain Dual Attention and Spectral Fusion Framework for Robust Image Captioning with Instruction-Tuned VLM Caption Refinement.**

## Overview

This repository presents a **novel image captioning architecture** that integrates **frequency-domain spectral attention**, **dual-branch visual reasoning**, and **instruction-tuned visual language model (VLM) refinement** for generating accurate, detailed, and grounded image captions.

The framework fuses **YOLOv8-based object detection** and **Xception-based scene classification** using **Discrete Cosine Transform (DCT)** in the frequency domain. It then applies a **Spectral Attention Module** using FFT to capture long-range dependencies efficiently, followed by a **GRU-based decoder** and an **LLM-guided refinement stage** for factual correctness.

---

## Key Features & Architecture
This framework improves upon classical CNN+RNN architectures, resulting in improved performance on the MSCOCO-2014 dataset.

### Key Contributions
* **Dual-Stream Encoder:** Object-level features from YOLOv8 + scene-level features from Xception.
* **Dual Bahdanau Attention:** Independent attention over object and scene embeddings.
* **Frequency-Domain Fusion (DCT):** Low-frequency spectral merging for compact and meaningful representations.
* **Spectral Attention Module (FFT-based):** Captures global feature interactions with reduced complexity.
* **GRU Caption Decoder:** Generates draft captions.
* **VLM Refinement:** Multimodal LLM (e.g., MiniGPT-4/LLaVA) improves caption factuality and grounding.
### Architecture Flow
1.  Input Image
2.  YOLOv8 + Xception Feature Extraction
3.  Dual Bahdanau Attention
4.  DCT-based Frequency Fusion
5.  FFT-based Spectral Attention
6.  GRU Decoder
7.  Instruction-Tuned VLM Caption Refinement

---

## Codebase & Notebooks
* **Main Project Code:** The main, runnable code for the entire project is located at:
    `src/improv_imgcap.py`

* **Colab Notebooks:** All experimental and development notebooks used for this project are located in the `colab_nb/` folder.

## Base Repository
This work builds upon and extends the baseline architecture from the following open-source repository:

[abdelhadie-amalla/image_captioning](https://github.com/abdelhadie-almalla/image_captioning)

Our framework enhances the original CNN–GRU pipeline with dual attention, frequency-domain fusion, spectral attention, and instruction-tuned VLM caption refinement.

---

## Quick Start
1.  Create a Python environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  The main execution script is `src/improv_imgcap.py`.

3.  Edit `configs/config.yaml` to point to your local dataset and weight locations.

---

## Usage
Run the main project stages using the provided scripts:
1.  **Run feature extraction:**
    ```bash
    bash scripts/extract_features.sh
    ```
2.  **Train the model:**
    ```bash
    bash scripts/train_model.sh
    ```
3.  **Evaluate the model:**
    ```bash
    bash scripts/evaluate_model.sh
    ```
4.  **Produce refined captions:**
    *(The VLM refinement wrapper uses the rule-based fallback by default)*
    ```bash
    bash scripts/refine_captions.sh
    ```

---
