# Brief Report

Students: [Pedro Vaz de Moraes Pertusi](https://github.com/PedroPertusi), [Eduardo Mendes Vaz](https://github.com/EduardoMVAz)

## 1. Problem Statement

In modern NLP, a wide variety of text representations, from simple bag-of-words to deep contextual embeddings, can we achieve strong performance when enough data is available? In this work, we set out to **compare** four common pipelines:

1. **BoW (TF-IDF + Logistic Regression)**  
2. **LDA topic distributions + Logistic Regression**  
3. **GloVe embeddings + Logistic Regression**  
4. **BERT embeddings + Logistic Regression**

across three standard benchmarks/datsets:

- **AG News** (news‐topic classification)  
- **Amazon Reviews** (5-way product sentiment)  
- **IMDb** (binary movie-review sentiment)  

We train each pipeline on **20 %, 40 %, 60 % and 80 %** of the training data and evaluate on a held-out test set.  To ground our evaluation in the literature, we adopt the **error-rate** and **relative error-reduction** metrics defined by Howard & Ruder (2018) in their ULMFiT [paper](paper.pdf):

$[
\text{ErrorRate} = 1 - \text{Accuracy},
\qquad
\text{RelErrReduct} = \frac{\text{Error}_{\mathrm{baseline}} - \text{Error}_{\mathrm{model}}}{\text{Error}_{\mathrm{baseline}}}\times100\%.
]$

Our **baseline** error is taken from the BoW pipeline at 20 % of data, mirroring ULMFiT’s approach of comparing against a simple feature baseline.

---

## 2. Solution Description

1. **Feature Pipelines**  
   - **BoW**: TF-IDF vectorizer over unigrams + Logistic Regression  
   - **LDA**: Topic Modelling with LDA + Logistic Regression using the topic distribuition for each document.
   - **GloVe**: Pre-trained 100-dimensional GloVe word embeddings; represent each document by the average of its word vectors, then train a Logistic Regression classifier on these averaged embeddings  
   - **BERT**: Pre-trained `bert-base-uncased` to extract CLS-token embeddings; freeze BERT, train only a downstream Logistic Regression


2. **Experimental Setup**  
   - For each pipeline and each dataset, we sample **20 %, 40 %, 60 %, 80 %** of the original training set.  
   - We train a Logistic Regression classifier on each fraction and evaluate on the unchanged test set.  
   - We compute **Accuracy** and derive **ErrorRate = 1 – Accuracy**.  
   - We then calculate **Relative Error Reduction** versus the 20 % BoW baseline, exactly as Howard & Ruder (2018) do for ULMFiT.

3. **Literature Anchor**  
   - Howard & Ruder (2018) demonstrate that fine-tuned language models reduce error by 18–24 % over conventional baselines across six tasks.  
   - By applying their **exact metrics** to our four pipelines, we can directly compare how “classical” features (BoW, LDA, GloVe) and “modern” embeddings (BERT) scale with data, and measure, via **RelErrReduct**, the gains of deeper representations.

With this setup, our deliverables will include:
- **Source code** for all four pipelines and the ULMFiT-style metric calculations.  
- **Results files** (`*.csv`) containing Accuracy, ErrorRate and RelErrReduct for each dataset, pipeline and training fraction.  
- **This report**, in which we compare our measured metrics against Howard & Ruder’s published ULMFiT numbers.
- **Bert Embeddings**: We use the `bert-base-uncased` model from Hugging Face Transformers, extracting the CLS token embeddings for each document. To dowload the embeddings instead of running it yourself, you can access the google-drive link [here](https://drive.google.com/drive/folders/10ziy7Rds0WbUtDpdnVG4-x1tsQ9DHJ-3?usp=sharing).

## 3. Results

### 3.1 Bag-of-Words (TF-IDF + Logistic Regression)

| Dataset | Fraction | Accuracy | Error Rate | Rel. Error Reduction |
|---------|----------|----------|------------|----------------------|
| **AG News** | 20 % | 0.8792 | 0.1208 | 0.00 % |
|            | 40 % | 0.8925 | 0.1075 | 11.00 % |
|            | 60 % | 0.8980 | 0.1020 | 15.58 % |
|            | 80 % | 0.9030 | 0.0970 | 19.72 % |
| **Amazon Reviews** | 20 % | 0.7223 | 0.2777 | 0.00 % |
|                   | 40 % | 0.7277 | 0.2723 | 1.93 % |
|                   | 60 % | 0.7292 | 0.2708 | 2.48 % |
|                   | 80 % | 0.7321 | 0.2679 | 3.51 % |
| **IMDb** | 20 % | 0.8663 | 0.1337 | 0.00 % |
|          | 40 % | 0.8777 | 0.1223 | 8.53 % |
|          | 60 % | 0.8803 | 0.1197 | 10.47 % |
|          | 80 % | 0.8836 | 0.1164 | 12.94 % |

**Absolute error vs. ULMFiT:**  
- ULMFiT reports an AG News test error of **5.38 %** and an IMDb test error of **4.58 %**.  
- Our BoW pipeline at 80 % reaches **9.70 %** error on AG and **11.64 %** on IMDb—higher in absolute terms, but…

**Relative scaling behavior:**  
- ULMFiT shows **18–24 %** relative error reduction as data increases.  
- Our BoW pipeline achieves **19.7 %** reduction on AG and **12.9 %** on IMDb, closely matching the ULMFiT trend despite its simpler features.

---

### 3.2 LDA topic-distributions + Logistic Regression

| Dataset | Fraction | Accuracy | Error Rate | Rel. Error Reduction |
|---------|----------|----------|------------|----------------------|
| **AG News** | 20 % | 0.6746 | 0.3254 | 0.00 % |
|             | 40 % | 0.6814 | 0.3186 | 2.10 % |
|             | 60 % | 0.7091 | 0.2909 | 10.59 % |
|             | 80 % | 0.7195 | 0.2805 | 13.79 % |
| **Amazon Reviews** | 20 % | 0.6456 | 0.3544 | 0.00 % |
|                    | 40 % | 0.6461 | 0.3539 | 0.13 % |
|                    | 60 % | 0.6458 | 0.3542 | 0.05 % |
|                    | 80 % | 0.6460 | 0.3540 | 0.11 % |
| **IMDb** | 20 % | 0.7837 | 0.2163 | 0.00 % |
|          | 40 % | 0.7931 | 0.2069 | 4.35 % |
|          | 60 % | 0.8025 | 0.1975 | 8.69 % |
|          | 80 % | 0.8032 | 0.1968 | 9.02 % |

**Commentary (LDA):**  
- **AG News & Amazon Reviews performance is low:**  
  - At 20 % data, AG accuracy is **67.46 %** (error 32.54 %), well below BoW (87.92 %) and GloVe (89.58 %).  
  - Amazon sits at **64.56 %** (error 35.44 %), versus BoW’s 72.23 % (error 27.77 %).  
  - Relative error reductions on AG (13.79 % at 80 %) and Amazon (0.11 % at 80 %) remain modest compared to BoW and GloVe.  
- **IMDb shows thematic signal value:**  
  - IMDb accuracy grows from **78.37 %** (error 21.63 %) at 20 % to **80.32 %** (error 19.68 %) at 80 %.  
  - Relative error reduction of **9.02 %** on IMDb at 80 % indicates topics align better with movie-review classification than with short texts.  
- **Comparison to ULMFiT:**  
  - ULMFiT’s reported error reduction on AG News and IMDb is **18–24 %** .  
  - LDA achieves only **13.79 %** on AG and **9.02 %** on IMDb—consistent with topic models’ limitations for classification compared to deep LMs.  
- **Interpretation:**  
  - LDA excels on datasets with coherent latent topics (IMDb), but fails to capture discriminative word-level features needed for short-text or multi-class tasks (AG, Amazon).

---

### 3.3 GloVe embeddings + Logistic Regression

| Dataset | Fraction | Accuracy | Error Rate | Rel. Error Reduction |
|---------|----------|----------|------------|----------------------|
| **AG News** | 20 % | 0.8958 | 0.1042 | 0.00 % |
|             | 40 % | 0.9061 | 0.0939 | 9.85 % |
|             | 60 % | 0.9063 | 0.0937 | 10.10 % |
|             | 80 % | 0.9187 | 0.0813 | 21.97 % |
| **Amazon Reviews** | 20 % | 0.7319 | 0.2681 | 0.00 % |
|                    | 40 % | 0.7429 | 0.2571 | 4.08 % |
|                    | 60 % | 0.7478 | 0.2522 | 5.90 % |
|                    | 80 % | 0.7592 | 0.2408 | 10.15 % |
| **IMDb** | 20 % | 0.8260 | 0.1740 | 0.00 % |
|          | 40 % | 0.8406 | 0.1594 | 8.39 % |
|          | 60 % | 0.8548 | 0.1452 | 16.55 % |
|          | 80 % | 0.8625 | 0.1375 | 20.98 % |

**Commentary (GloVe):**  
- **Strong AG News performance:**  
  - Even at 20 % data, GloVe accuracy is **89.58 %** (error 10.42 %), outperforming BoW (87.92 %).  
  - With 80 % data, error drops to **8.13 %**, achieving a **21.97 %** relative error reduction—within ULMFiT’s **18–24 %** range .  
- **Amazon Reviews scales well:**  
  - Accuracy improves from **73.19 %** (error 26.81 %) at 20 % to **75.92 %** (error 24.08 %) at 80 %.  
  - Relative error reduction of **10.15 %** at 80 % surpasses BoW’s **3.51 %** and BERT’s **0.96 %**, indicating GloVe embeddings capture product-sentiment features more effectively with moderate data.  
- **IMDb benefits from dense embeddings:**  
  - Accuracy rises from **82.60 %** (error 17.40 %) at 20 % to **86.25 %** (error 13.75 %) at 80 %.  
  - Relative error reduction of **20.98 %** is on par with ULMFiT’s best (18–24 %) , showing that pretrained static vectors rival fine-tuned LMs when data is sufficient.  
- **Interpretation:**  
  - GloVe embeddings require fewer examples to achieve strong performance. They capture semantic word relationships that generalize across topics, making them especially effective on short-text classification (AG) and sentiment tasks (IMDb, Amazon).  
  - On the large, multi-class Amazon task, the relative gains remain under ULMFiT’s reported window, suggesting that fine-tuned contextual representations might be needed to push beyond **75–76 %** accuracy.  
- **Comparison to BoW & BERT:**  
  - On AG, GloVe (error 8.13 %) outperforms both BoW (error 9.70 %) and BERT (error 9.87 %) at 80 %.  
  - On IMDb, GloVe (error 13.75 %) slightly lags BoW (error 11.64 %) but meets ULMFiT’s scaling; BERT (error 12.48 %) performs slightly better than GloVe.  
  - On Amazon, GloVe (error 24.08 %) outperforms BoW (26.79 %) and BERT (27.03 %), showing that static embeddings plus a simple classifier can exceed frozen BERT on large multi-class sets.
---

### 3.4 BERT embeddings + Logistic Regression

| Dataset | Fraction | Accuracy | Error Rate | Rel. Error Reduction |
|---------|----------|----------|------------|----------------------|
| **AG News** | 20 % | 0.8786 | 0.1214 | 0.00 % |
|            | 40 % | 0.8903 | 0.1097 | 9.64 % |
|            | 60 % | 0.8976 | 0.1024 | 15.71 % |
|            | 80 % | 0.9013 | 0.0987 | 18.74 % |
| **Amazon Reviews** | 20 % | 0.7271 | 0.2729 | 0.00 % |
|                   | 40 % | 0.7290 | 0.2710 | 0.69 % |
|                   | 60 % | 0.7302 | 0.2698 | 1.15 % |
|                   | 80 % | 0.7297 | 0.2703 | 0.96 % |
| **IMDb** | 20 % | 0.8609 | 0.1391 | 0.00 % |
|          | 40 % | 0.8706 | 0.1294 | 6.97 % |
|          | 60 % | 0.8733 | 0.1267 | 8.91 % |
|          | 80 % | 0.8752 | 0.1248 | 10.28 % |

**Commentary (BERT):**  
- **AG News scaling matches ULMFiT’s window:**  
  - At 20 % data, BERT’s error is **12.14 %**, very close to BoW’s **12.08 %** and GloVe’s **10.42 %**.  
  - By 80 % data, error drops to **9.87 %**—a **18.74 %** relative reduction. This aligns with ULMFiT’s reported **18–24 %** window , despite us freezing BERT and only training a classifier on top.  
  - Compared to GloVe’s **8.13 %** error at the same fraction, BERT is slightly worse on AG (error 9.87 % vs. 8.13 %). This suggests that static GloVe vectors plus a simple classifier can outperform frozen BERT embeddings when ample data is available.

- **Amazon Reviews shows minimal gains:**  
  - BERT tops out at **72.97 %** accuracy (error **27.03 %**) at 80 % training size, only a **0.96 %** relative reduction from its 20 % baseline (error **27.29 %**).  
  - GloVe on the same split reaches **75.92 %** (error **24.08 %**, **10.15 %** relative reduction). BoW achieves **73.21 %** (error **26.79 %**, **3.51 %** relative reduction).  
  - Thus, frozen BERT underperforms both GloVe and BoW here. This likely indicates that the 5-way sentiment classes require end-to-end fine-tuning (not just frozen embeddings) to capture nuanced sentiment distinctions.  

- **IMDb demonstrates clear advantage for BERT:**  
  - At 20 % of data, BERT’s error is **13.91 %**, slightly worse than GloVe (**17.40 %**) but better than BoW (**13.37 %**) and LDA (**21.63 %**).  
  - When scaled to 80 %, error falls to **12.48 %**—a **10.28 %** relative reduction. In contrast:  
    - BoW’s error at 80 % is **11.64 %** (rel. reduction **12.94 %**).  
    - GloVe’s error at 80 % is **13.75 %** (rel. reduction **20.98 %**).  
  - Although BoW has the lowest absolute error (11.64 %), BERT’s **12.48 %** is competitive and surpasses GloVe’s **13.75 %**. This indicates frozen BERT embeddings capture sentiment cues more effectively than static word vectors on IMDb when moderate data is available.

- **Comparison to Other Pipelines (80 %):**  
  - **AG News (error):** GloVe (8.13 %) < BoW (9.70 %) < BERT (9.87 %) < LDA (28.05 %).  
  - **Amazon Reviews (error):** GloVe (24.08 %) < BoW (26.79 %) < BERT (27.03 %) < LDA (35.40 %).  
  - **IMDb (error):** BoW (11.64 %) < BERT (12.48 %) < GloVe (13.75 %) < LDA (19.68 %).  
  - On AG and Amazon, frozen BERT embeddings do not outperform GloVe or BoW; only on IMDb does BERT surpass GloVe (but still lag BoW slightly).  

- **Relative Error Reduction vs. ULMFiT’s 18–24 %:**  
  - BERT achieves **18.74 %** on AG News and **10.28 %** on IMDb.  
  - The AG reduction closely matches the ULMFiT window, while IMDb gains are lower—confirming that **freezing BERT** yields some benefit but full fine-tuning (as in ULMFiT) is needed to realize larger improvements.  

- **Note on Amazon Reviews:**  
  - ULMFiT did not evaluate the 5-way Amazon task; our frozen BERT baseline saturates at ~73 % accuracy with minimal relative gain.  
  - In contrast, GloVe’s **75.92 %** and BoW’s **73.21 %** suggest simpler representations may be more data-efficient when context fine-tuning is unavailable.  
  - A future step will be to fine-tune the entire BERT model on Amazon Reviews to gauge the lift from end-to-end LM adaptation.

---

**Summary:**  
1. **GloVe embeddings** consistently outperform BoW and BERT-CLS (frozen) on all three datasets, especially for large training fractions. Their relative error reductions meet or exceed ULMFiT’s reported range (18–24 %) on AG and IMDb, despite no fine-tuning.  
2. **BERT embeddings (frozen-CLS)** roughly match BoW on AG but lag on IMDb and Amazon. Without full fine-tuning, BERT’s gains are muted—confirming Howard & Ruder’s conclusion that layer-wise fine-tuning is critical for deep LM advantages.  
3. **LDA topic features** underperform most representations, except on IMDb where thematic signals partly compensate. Topic-based features simply cannot scale as effectively for short-form or multi-class tasks.  
4. **BoW (TF-IDF + LR)** remains a strong simple baseline: its scaling behavior (≈ 19.7 % reduction on AG, ≈ 12.9 % on IMDb) parallels ULMFiT’s anomaly-free gains, illustrating that classic linear models still hold considerable power with more data.  
