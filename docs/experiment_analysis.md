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

The model also provides attention weights for later inspection, offering an additional diagnostic view of learned token relationships that is not available in the recurrent baselines. These weights are useful for inspecting attention patterns, but they should not be interpreted as complete explanations of the model's predictions.

---

## 6. Training Loss and Generalization

The neural results make the difference between **fitting the training data** and **generalizing to unseen data** especially visible.

| Model | Final Training Loss | Test F1 |
|---|---:|---:|
| BiLSTM + FastText | **0.0390** | 0.7813 |
| Self-Attention | 0.2022 | **0.8601** |
| LSTM | 0.4207 | 0.7906 |
| Vanilla RNN | 0.6976 | 0.4874 |

If training loss alone were used to select the model, BiLSTM + FastText would appear overwhelmingly superior.

The held-out test result tells a different story.

Self-Attention finished with a substantially higher training loss (`0.2022`) but achieved much better test F1 (`0.8601`).

This is why the experiment needs both types of evidence:

```text
Training loss
     ↓
How strongly did the model fit the training objective?

Held-out metrics
     ↓
How well did that learned behavior transfer to unseen reviews?
```

The BiLSTM result is consistent with overfitting or, more cautiously, a **generalization gap**. Because the experiment did not maintain a dedicated validation-loss curve for model selection, the exact onset and magnitude of overfitting cannot be established precisely from the available evidence.

The defensible conclusion is therefore:

> The BiLSTM + FastText model fit the training data much more strongly than the other neural models, but this did not translate into better held-out performance.

---

## 7. Classical vs Neural Models

Combining the reference results gives:

| Model | Family | F1 Score |
|---|---|---:|
| **LinearSVC** | Classical | **0.8750** |
| Logistic Regression | Classical | 0.8748 |
| Self-Attention | Neural | 0.8601 |
| Random Forest | Classical | 0.8544 |
| Bernoulli Naive Bayes | Classical | 0.8117 |
| LSTM | Neural | 0.7906 |
| BiLSTM + FastText | Neural | 0.7813 |
| Vanilla RNN | Neural | 0.4874 |

### Complexity did not automatically win

The strongest result in the complete comparison came from LinearSVC, not from the largest or most sophisticated neural architecture.

Logistic Regression was essentially tied with it.

This is an important result because sparse bag-of-words-style representations can be extremely effective for sentiment classification when the presence of discriminative words and phrases already carries strong predictive information.

A neural model may learn richer sequence-dependent representations, but that additional representational power is useful only if it translates into better generalization for the task and training setup.

### Self-Attention came closest

Self-Attention was the only neural model that approached the strongest classical baselines:

```text
LinearSVC F1          : 0.8750
Logistic Regression F1: 0.8748
Self-Attention F1     : 0.8601
```

The gap between Self-Attention and LinearSVC was `0.0149`.

That is a much smaller gap than the one between the linear models and the recurrent neural baselines.

### Practical model choice

If the decision were based only on the evidence generated in these experiments, LinearSVC would be a strong practical candidate because it combines:

- the highest observed held-out F1
- relatively low architectural complexity
- efficient inference
- straightforward deployment
- no GPU requirement for inference
- simpler operational maintenance than the neural alternatives

This does **not** make the neural experiments unnecessary.

The neural models expose different learning mechanisms and provide useful evidence about recurrent memory, pretrained embeddings, bidirectional context, attention, model capacity, and generalization.

The best engineering choice and the most educational model are not necessarily the same thing.

---

## 8. Error Analysis

Aggregate metrics show how often the model is correct. Error analysis helps explain **why** it is wrong.

The strongest overall model, LinearSVC, was inspected through false-positive and false-negative examples.

### Mixed sentiment

Several errors involved reviews that contain both positive and negative language.

Some negative reviews contained locally positive expressions such as:

```text
"worth the entertainment value"
"entertaining"
"decent film"
```

while the overall review remained critical.

A sparse text classifier can learn that these terms correlate with positive sentiment without fully representing how the surrounding review changes their meaning.

The reverse problem also occurs: a positive review may discuss dark, unpleasant, violent, or otherwise negative subject matter while the reviewer is actually praising the film.

### False positives

A false positive occurs when:

```text
Actual sentiment    : Negative
Predicted sentiment : Positive
```

Mixed reviews are a natural source of this error. Positive local vocabulary can dominate the feature representation even when the writer's final judgement is negative.

### False negatives

A false negative occurs when:

```text
Actual sentiment    : Positive
Predicted sentiment : Negative
```

This can happen when the review contains many negative-sounding words because of the film's subject matter, criticism of particular elements, or a contrastive writing style, even though the overall judgement is favorable.

### What the errors reveal

The errors highlight a limitation shared in different ways by several models:

> **Detecting sentiment-bearing words is not the same as understanding the sentiment of the complete review.**

Long reviews can contain:

- contrast
- qualification
- mixed praise and criticism
- narrative description
- sarcasm
- sentiment shifts
- negative subject matter described positively

The model must eventually compress all of that into one binary decision.

---

## 9. Attention Analysis

The Self-Attention experiment also inspected attention weights for selected reviews.

The implemented model prepends a learnable classification token (`CLS`) and uses the resulting `CLS` representation for sentiment prediction. The attention weights provide a useful way to inspect which token relationships received relatively greater emphasis inside the attention mechanism.

### Correct Classifications

In some correctly classified examples, higher attention weights appeared around evaluative or sentiment-bearing expressions.

This was especially visible when strongly positive or negative words were consistent with the overall sentiment of the review.

These patterns are useful as diagnostic evidence because they show that semantically relevant parts of the review can receive stronger attention during processing.

### Misclassifications

Misclassified examples also revealed an important limitation.

Some reviews contained locally positive or negative expressions that did not represent the overall sentiment of the complete review.

In such cases, relatively strong attention could appear around these local sentiment-bearing expressions even though the broader review expressed a different conclusion.

This mirrors the mixed-sentiment problem observed during the classical-model error analysis.

Reviews containing contrast, mixed praise and criticism, sarcasm, or sentiment shifts can therefore remain difficult even when the model identifies locally meaningful relationships.

### Attention Is Evidence, Not a Complete Explanation

A high attention weight tells us that a token relationship received greater emphasis within the model's attention computation.

It does not by itself prove:

```text
"This word caused the prediction."
```

or:

```text
"The model understands this word exactly as a person would."
```

The final prediction emerges from the complete learned representation rather than from the attention weights alone.

In this implementation, the prediction is produced through:

```text
Token Embeddings
       +
Positional Information
        ↓
Multi-Head Self-Attention
        ↓
Attention-Transformed Representations
        ↓
CLS Representation
        ↓
Normalization
        ↓
Classification Layers
        ↓
Sentiment Prediction
```

The learned token embeddings, positional information, attention-transformed representations, aggregated `CLS` representation, and downstream classification layers all contribute to the final output.

Attention visualization is therefore best treated as a **diagnostic and interpretability aid**. It can help identify patterns worth investigating and generate hypotheses about model behavior, but it should not be treated as a complete causal explanation of why a particular prediction was made.

### Main Takeaway

The attention analysis provides two useful lessons:

1. Self-Attention can model relationships between different positions in a review without relying on recurrent information flow.
2. Attention weights can help inspect model behavior, but they must be interpreted together with the complete learned representation and the overall review context.

---

## 10. What the Experiments Demonstrate

The completed experiments support several useful conclusions.

### Strong classical baselines matter

LinearSVC and Logistic Regression achieved the two strongest held-out F1 scores.

A sophisticated neural architecture should therefore be compared against strong classical baselines rather than assumed to be superior because it is newer or more complex.

### LSTM gating made a large practical difference

Under the implemented configuration, moving from Vanilla RNN to LSTM improved F1 from `0.4874` to `0.7906`.

The training curves also changed from near-stagnation to sustained learning.

### Training fit and generalization are different

BiLSTM + FastText achieved the lowest training loss by a large margin but did not produce the strongest test result.

The experiment gives a concrete example of why training loss cannot be used as the sole model-selection criterion.

### More parameters did not guarantee better performance

BiLSTM + FastText had the largest trainable parameter count, yet its held-out F1 was below the simpler LSTM and far below Self-Attention.

Capacity is useful only when it is converted into generalizable behavior.

### Self-Attention was the strongest neural architecture

Self-Attention reached `0.8601` F1 and was the only neural model to approach the strongest classical baselines.

### Model choice is an engineering decision

A final model should not be selected by F1 alone.

Relevant considerations include:

- held-out predictive quality
- stability across runs
- training cost
- inference cost
- model size
- hardware requirements
- interpretability needs
- operational complexity
- maintainability

For the evidence available in this project, LinearSVC offers a particularly strong balance.

---

## 11. What the Experiments Do Not Establish

A useful experiment analysis should also state what cannot be concluded.

### The neural architectures were not exhaustively tuned

The neural experiments were designed to compare modeling approaches, not to find the maximum achievable score for every architecture.

Therefore, the results should not be interpreted as:

> "LSTM can only achieve 0.7906 F1 on IMDb."

They show what the implemented LSTM configuration achieved in this experiment.

### BiLSTM and FastText effects cannot be separated

Bidirectionality and pretrained embeddings were introduced together.

The result therefore cannot tell us whether the observed behavior came primarily from:

- bidirectionality
- FastText
- the combination
- the larger parameter count
- interactions with the training configuration

A controlled ablation is needed.

### The neural test set is not a validation set

The neural workflow does not include a dedicated validation split for decisions such as epoch selection, learning-rate tuning, regularization, or early stopping.

A more rigorous model-development workflow would reserve validation data for those decisions and use the test set only once for final evaluation.

### One run does not measure neural variance

Neural training is stochastic.

The reference scores describe completed runs, not a distribution across random seeds.

Repeated experiments would be needed to estimate mean performance and variance.

### Attention weights are not causal explanations

The attention visualization provides insight into model behavior but does not establish that highly weighted tokens alone caused a prediction.

### The comparison does not prove a universal ranking

The results are specific to:

- this dataset
- these representations
- these model implementations
- these hyperparameters
- these training procedures

They do not imply that LinearSVC universally outperforms neural sentiment models or that Self-Attention universally outperforms LSTM.

---

## 12. Useful Follow-Up Experiments

The existing results suggest several controlled experiments that would deepen the analysis without changing the purpose of v1.

### Separate bidirectionality from FastText

A useful ablation matrix would be:

| Recurrent Direction | Embeddings |
|---|---|
| LSTM | learned/random |
| BiLSTM | learned/random |
| LSTM | FastText |
| BiLSTM | FastText |

This would allow the effect of bidirectionality and pretrained embeddings to be measured independently.

### Add validation-based early stopping

Track training and validation loss together and stop when validation performance no longer improves.

This would be especially useful for investigating the BiLSTM generalization gap.

### Tune neural regularization

Potential variables include:

- dropout
- weight decay
- learning rate
- learning-rate schedule
- hidden dimension
- number of layers

### Compare sequence lengths

The current maximum sequence length is `400`.

Testing shorter and longer limits could reveal the trade-off between:

- retained context
- compute cost
- recurrent difficulty
- truncation loss

### Repeat neural runs across random seeds

Rather than comparing one score per model, repeated runs could report:

```text
mean F1 ± standard deviation
```

This would make the neural comparison statistically more informative.

### Add stronger representation baselines

A later experiment could compare `CountVectorizer` with alternatives such as TF-IDF while keeping the classifier fixed.

That would isolate the effect of representation from classifier choice.

These are follow-up experiments, not missing requirements for the current repository.

---

## 13. Key Learnings

The most useful outcome of the project is not one winning score. It is the set of modeling lessons exposed by the comparison.

1. **Always establish strong simple baselines.**  
   LinearSVC and Logistic Regression remained stronger than all neural models in the completed comparison.

2. **Accuracy alone is not enough.**  
   Precision, recall, F1, confusion matrices, and class-specific error behavior reveal different aspects of a classifier.

3. **Cross-validation and held-out testing serve different purposes.**  
   Model selection should happen before final test evaluation.

4. **Vanilla recurrence can struggle on long sequences.**  
   The RNN's near-flat loss and chance-level performance made this limitation visible in practice.

5. **Gating can materially improve recurrent learning.**  
   The LSTM showed a large improvement over the basic RNN under the same task.

6. **Lower training loss is not the same as better generalization.**  
   BiLSTM + FastText provided the clearest example.

7. **More parameters do not guarantee better results.**  
   The largest neural model was not the strongest held-out classifier.

8. **Attention can improve contextual modeling without solving every semantic problem.**  
   Self-Attention was the strongest neural model, yet mixed-sentiment reviews remained challenging.

9. **Interpretability tools should be used carefully.**  
   Attention weights can help inspect model behavior but are not complete causal explanations.

10. **Model selection is ultimately an engineering trade-off.**  
    Predictive performance has to be considered together with compute, inference requirements, complexity, and maintainability.

---

## Final Perspective

The experiment started as a comparison of sentiment-classification models, but the most valuable result is broader.

It demonstrates why machine-learning evaluation should move beyond the question:

> **Which model is the most sophisticated?**

toward:

> **Which model learned useful behavior, generalized to unseen data, and provides an appropriate trade-off for the problem we are actually solving?**

For this experiment, the answer is nuanced:

- **LinearSVC** produced the strongest observed held-out F1.
- **Logistic Regression** was essentially tied while remaining very efficient.
- **Self-Attention** was the strongest neural model.
- **LSTM** demonstrated the practical value of gated recurrence over a basic RNN.
- **BiLSTM + FastText** demonstrated that an extremely low training loss can coexist with weaker held-out generalization.
- **Vanilla RNN** provided a useful failure case showing that architectural simplicity is not always sufficient for long-sequence learning.

Together, those outcomes provide a more useful learning record than a single winning metric.

---

**Related documentation**

- [Project README](../README.md)
- [Architecture Documentation](architecture/README.md)
