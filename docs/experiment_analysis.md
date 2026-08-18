# Experiment Analysis

This document preserves the deeper machine-learning analysis behind the IMDb Sentiment Classification project.

The root [`README.md`](../README.md) is the project entry point: it explains the purpose, usage, headline results, and current scope. The [`docs/architecture/README.md`](architecture/README.md) explains how the modular codebase is structured.

This document focuses on a different question:

> **What did the experiments actually show, how should the results be interpreted, and what can — and cannot — be concluded from them?**

The analysis is based on the completed classical experiment and the completed neural-network notebook experiment that formed the basis of the modular implementation.

---

## Table of Contents

- [1. Experimental Objective](#1-experimental-objective)
- [2. Experimental Setup](#2-experimental-setup)
- [3. Interpreting the Evaluation Metrics](#3-interpreting-the-evaluation-metrics)
- [4. Classical Machine-Learning Analysis](#4-classical-machine-learning-analysis)
- [5. Neural-Network Analysis](#5-neural-network-analysis)
- [6. Training Loss and Generalization](#6-training-loss-and-generalization)
- [7. Classical vs Neural Models](#7-classical-vs-neural-models)
- [8. Error Analysis](#8-error-analysis)
- [9. Attention Analysis](#9-attention-analysis)
- [10. What the Experiments Demonstrate](#10-what-the-experiments-demonstrate)
- [11. What the Experiments Do Not Establish](#11-what-the-experiments-do-not-establish)
- [12. Useful Follow-Up Experiments](#12-useful-follow-up-experiments)
- [13. Key Learnings](#13-key-learnings)

---

## 1. Experimental Objective

The project compares several ways of approaching the same binary sentiment-classification problem.

The model families range from sparse linear classifiers to recurrent neural networks and self-attention:

| Family | Models |
|---|---|
| Classical ML | Logistic Regression, Bernoulli Naive Bayes, LinearSVC, Random Forest |
| Neural Networks | Vanilla RNN, LSTM, BiLSTM + FastText, Self-Attention |

The purpose of the comparison is broader than finding the largest F1 score.

The experiments examine:

- how strong classical text-classification baselines can be
- whether recurrent memory improves performance on long reviews
- what happens when bidirectionality and pretrained embeddings are introduced
- how self-attention compares with recurrent architectures
- whether lower training loss translates into better held-out performance
- how model complexity relates to predictive performance
- what kinds of reviews remain difficult to classify

This makes the project a comparison of **model behavior and trade-offs**, not simply a leaderboard.

---

## 2. Experimental Setup

### Dataset

The experiments use the Stanford IMDb Large Movie Review Dataset:

| Split | Reviews | Positive | Negative |
|---|---:|---:|---:|
| Training | 25,000 | 12,500 | 12,500 |
| Test | 25,000 | 12,500 | 12,500 |

The balanced class distribution is important when interpreting accuracy: a classifier near `0.50` accuracy is effectively close to chance performance on this task.

### Classical representation

The classical workflow cleans the review text and uses `CountVectorizer` inside a scikit-learn `Pipeline`.

Model selection uses `GridSearchCV`, optimized by F1 score. Keeping vectorization inside the pipeline ensures that the vectorizer is fitted independently within each cross-validation training fold.

The selected estimator is then evaluated on the held-out IMDb test split.

### Neural representation

The neural workflow:

1. tokenizes the reviews
2. builds a vocabulary from the training data
3. converts tokens to integer IDs
4. pads or truncates sequences to a maximum length of `400`
5. batches the sequences through PyTorch `Dataset` and `DataLoader`

The standard batch size is `64`.

The completed reference notebook trained the neural architectures for `10` epochs.

### A note on comparability

The classical and neural experiments use the same sentiment-classification task and underlying IMDb split, but they do **not** differ only in classifier architecture.

Classical models use sparse vectorized text. Neural models use learned token embeddings and sequence processing.

The comparison is therefore useful as a practical model-family comparison, but it should not be interpreted as a perfectly controlled experiment in which only one variable changes.

---

## 3. Interpreting the Evaluation Metrics

Four held-out metrics are used throughout the project.

### Accuracy

Accuracy measures the fraction of all reviews classified correctly.

For this balanced dataset it is easy to interpret, but it does not show whether errors are distributed differently between positive and negative reviews.

### Precision

For the positive class, precision asks:

> Of the reviews predicted as positive, how many were actually positive?

High precision means relatively few negative reviews were incorrectly labelled positive.

### Recall

Recall asks:

> Of the reviews that were actually positive, how many did the model identify?

High recall means relatively few positive reviews were missed.

### F1 score

F1 is the harmonic mean of precision and recall.

It is used as the primary comparison metric in this project because it rewards models that balance both quantities rather than optimizing one while sacrificing the other.

### Confusion matrix

The confusion matrix adds information that aggregate scores cannot show directly:

```text
                 Predicted
               Negative Positive
Actual Negative    TN      FP
Actual Positive    FN      TP
```

It is especially useful when inspecting whether a model has developed a strong preference for one class.

### Cross-validation F1

For classical models, `GridSearchCV` provides an additional model-selection signal before the final held-out test evaluation.

The important distinction is:

```text
Cross-validation F1 → model / hyperparameter selection
Test F1             → final held-out evaluation
```

The test set should not be treated as the search mechanism itself.

---

## 4. Classical Machine-Learning Analysis

### Overall results

| Model | Accuracy | Precision | Recall | F1 Score | CV F1 | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| **LinearSVC** | **0.8746** | **0.8719** | 0.8782 | **0.8750** | 0.8615 | 3.60 min |
| **Logistic Regression** | 0.8734 | 0.8658 | **0.8839** | 0.8748 | **0.8663** | **1.46 min** |
| **Random Forest** | 0.8530 | 0.8463 | 0.8627 | 0.8544 | 0.8519 | 54.16 min |
| **Bernoulli Naive Bayes** | 0.8232 | 0.8683 | 0.7619 | 0.8117 | 0.7985 | 1.59 min |

Two models stand out immediately: LinearSVC and Logistic Regression.

Their held-out F1 scores differ by only `0.0002`. That difference is too small to support a strong claim that one is meaningfully better from this run alone. The more useful conclusion is that **both linear classifiers are extremely competitive on this sparse text representation**.

### Logistic Regression

Logistic Regression achieved:

```text
Accuracy : 0.8734
Precision: 0.8658
Recall   : 0.8839
F1       : 0.8748
CV F1    : 0.8663
Runtime  : 1.46 min
```

Its most notable result is the highest recall among the classical models.

It also achieved the highest cross-validation F1 and the shortest reported search runtime in the full classical comparison.

This makes Logistic Regression a particularly strong baseline: its predictive result is essentially tied with LinearSVC while remaining computationally efficient and providing probabilistic outputs if those are needed in a later application.

### LinearSVC

LinearSVC achieved:

```text
Accuracy : 0.8746
Precision: 0.8719
Recall   : 0.8782
F1       : 0.8750
CV F1    : 0.8615
Runtime  : 3.60 min
```

It produced the highest held-out F1 in the complete experiment.

The important point is not the tiny `0.0002` advantage over Logistic Regression. It is that a margin-based linear model operating on sparse text features remained competitive with — and in this experiment slightly exceeded — all of the more complex neural architectures.

For a practical model-selection decision based on these experiments alone, LinearSVC would therefore be a strong candidate.

### Bernoulli Naive Bayes

Bernoulli Naive Bayes achieved:

```text
Accuracy : 0.8232
Precision: 0.8683
Recall   : 0.7619
F1       : 0.8117
CV F1    : 0.7985
Runtime  : 1.59 min
```

Its precision remained relatively strong, but recall dropped substantially.

That means the model was comparatively conservative about predicting the positive class: when it predicted positive it was often correct, but it failed to recover a larger share of the actual positive reviews.

This precision-recall imbalance explains why its F1 score is clearly below the two strongest linear models.

### Random Forest

Random Forest achieved:

```text
Accuracy : 0.8530
Precision: 0.8463
Recall   : 0.8627
F1       : 0.8544
CV F1    : 0.8519
Runtime  : 54.16 min
```

Its F1 is respectable, but the runtime result is particularly important.

The reported hyperparameter-search runtime was approximately `54.16` minutes, compared with `1.46` minutes for Logistic Regression and `3.60` minutes for LinearSVC.

The additional computational cost did not produce a corresponding predictive improvement.

For this representation and task, that makes Random Forest a weaker engineering trade-off than the two linear alternatives.

### Cross-validation vs held-out performance

| Model | Best CV F1 | Test F1 |
|---|---:|---:|
| Logistic Regression | **0.8663** | 0.8748 |
| LinearSVC | 0.8615 | **0.8750** |
| Random Forest | 0.8519 | 0.8544 |
| Bernoulli Naive Bayes | 0.7985 | 0.8117 |

The held-out scores remain reasonably close to the cross-validation scores.

This is useful because the final comparison is not based solely on the test set. Model selection happened through cross-validation first, and the selected estimators were then evaluated against held-out data.

The small differences between CV and test performance are not evidence that the test set was used incorrectly; some variation between estimates is expected.

---

## 5. Neural-Network Analysis

### Overall results

| Model | Accuracy | Precision | Recall | F1 Score | Trainable Parameters | Final Training Loss |
|---|---:|---:|---:|---:|---:|---:|
| **Self-Attention** | **0.8629** | **0.8782** | **0.8427** | **0.8601** | 3,802,753 | 0.2022 |
| **LSTM** | 0.7903 | 0.7897 | 0.7914 | 0.7906 | 4,649,601 | 0.4207 |
| **BiLSTM + FastText** | 0.7959 | 0.8416 | 0.7291 | 0.7813 | 11,588,229 | **0.0390** |
| **Vanilla RNN** | 0.4968 | 0.4967 | 0.4785 | 0.4874 | 3,958,401 | 0.6976 |

The neural results reveal several distinct behaviors rather than one simple progression where every more sophisticated model performs better.

### Vanilla RNN: training largely stalled

The Vanilla RNN remained close to chance performance:

```text
Accuracy : 0.4968
F1       : 0.4874
```

Its loss changed only slightly across the ten epochs:

```text
Epoch 1  : 0.7007
Epoch 10 : 0.6976
```

The loss effectively plateaued instead of showing sustained convergence.

The reviews are processed as sequences of up to `400` tokens. In a vanilla recurrent network, useful information and gradients must propagate through many sequential steps.

The result is consistent with the practical difficulty basic RNNs have in retaining useful long-range information over long sequences.

This experiment therefore provides a useful baseline: the recurrent structure alone was not sufficient to learn the task effectively under the implemented configuration.

### LSTM: a major improvement over the basic RNN

The LSTM achieved:

```text
Accuracy : 0.7903
Precision: 0.7897
Recall   : 0.7914
F1       : 0.7906
```

Compared with the Vanilla RNN:

```text
Vanilla RNN F1 : 0.4874
LSTM F1        : 0.7906
Difference     : +0.3032
```

Its loss also behaved very differently:

```text
Epoch 1  : 0.6934
Epoch 10 : 0.4207
```

Unlike the RNN, the LSTM continued learning across the training period.

This provides a practical demonstration of the benefit of gated recurrence for long text sequences. The LSTM's cell state and gates provide mechanisms for retaining, updating, and forgetting information more effectively than the simple recurrent state.

The experiment does not prove that LSTMs will always outperform vanilla RNNs by this margin, but within this implementation the difference is substantial.

### BiLSTM + FastText: excellent training fit, weaker generalization

The BiLSTM experiment introduced two changes together:

1. bidirectional recurrence
2. pretrained 300-dimensional FastText embeddings

Its held-out result was:

```text
Accuracy : 0.7959
Precision: 0.8416
Recall   : 0.7291
F1       : 0.7813
```

Its training loss, however, fell dramatically:

```text
Epoch 1  : 0.5639
Epoch 5  : 0.1310
Epoch 10 : 0.0390
```

The model learned the training data extremely well.

Yet its test F1 (`0.7813`) was slightly below the simpler LSTM (`0.7906`).

This is one of the most important findings in the experiment:

> **A lower training loss does not imply better generalization.**

The BiLSTM + FastText model also had approximately `11.6 million` trainable parameters, substantially more than the other neural architectures.

Its greater capacity and pretrained representation allowed a much stronger fit to the training data, but that fit did not translate into the strongest held-out result.

The precision-recall pattern is also interesting:

```text
Precision : 0.8416
Recall    : 0.7291
```

The model was considerably better at being correct when it predicted the positive class than at recovering all positive reviews.

### Self-Attention: strongest neural result

The Self-Attention model achieved:

```text
Accuracy : 0.8629
Precision: 0.8782
Recall   : 0.8427
F1       : 0.8601
```

Its training loss decreased consistently:

```text
Epoch 1  : 0.5840
Epoch 5  : 0.2888
Epoch 10 : 0.2022
```

It substantially outperformed the Vanilla RNN, LSTM, and BiLSTM + FastText on held-out F1.

Unlike recurrent models, self-attention allows positions in the review to interact directly rather than requiring information to move sequentially through every intermediate timestep.

The implemented architecture uses:

```text
Token embeddings
      +
Positional encoding
      ↓
LayerNorm + Dropout
      ↓
Learnable CLS token
      ↓
Multi-Head Self-Attention
      ↓
CLS representation
      ↓
Classification head
```

The repository uses the term **Self-Attention** because the queries, keys, and values are derived from the same sequence.

The model also provides attention weights for later inspection, which creates an additional interpretability tool not present in the recurrent baselines.

---

## 6. Training Loss and Generalization

The neural results make the difference between **fitting the training data** and **generalizing to unseen data** especially visible.

| Model | Final Training Loss | Test F1 |
|---|---:|---:|
| BiLSTM + FastText | **0.0390** | 0.7813 |
| Self-Attention | 0.2022 | **0.8601** |
| LSTM | 0.4207 | 0.7906 |
| Vanilla RNN | 0.6976