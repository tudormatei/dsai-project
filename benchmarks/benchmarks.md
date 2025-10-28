# NO RULE BASED

## All 3 concepts

### CNN Results

```txt
Evaluating model...

=== RESULTS ===

--- Classification Concepts (Discrete) ---
Periodicity - Accuracy: 0.6316, AUROC: 0.8172
Temporal Stability - Accuracy: 0.7632, AUROC: 0.8819
Coordination - Accuracy: 0.7105, AUROC: 0.8785

--- Overall Performance ---
Overall Average Accuracy (classification concepts): 70.2%
Overall Average AUROC (classification concepts): 0.8592
```

### Logistic Regression

```txt
=== Logistic Regression (trained on GT concepts from TRAIN split) ===
Accuracy: 0.3684210526315789
AUROC: 0.8474282354070661
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         6
           1       0.25      0.29      0.27         7
           2       0.67      1.00      0.80         6
           3       0.43      0.50      0.46         6
           4       0.38      0.50      0.43         6
           5       0.00      0.00      0.00         7

    accuracy                           0.37        38
   macro avg       0.29      0.38      0.33        38
weighted avg       0.28      0.37      0.32        38

[[0 0 0 2 3 1]
 [0 2 0 1 1 3]
 [0 0 6 0 0 0]
 [0 0 3 3 0 0]
 [0 0 0 1 3 2]
 [0 6 0 0 1 0]]
```

### Full CBM Results

```txt
=== CNN Results ===

Classification Concepts (Discrete)
Periodicity - Accuracy: 0.6053, AUROC: 0.7906
Temporal Stability - Accuracy: 0.7368, AUROC: 0.8663
Coordination - Accuracy: 0.7105, AUROC: 0.8377

Overall Performance
Overall Average Accuracy (classification concepts): 68.4%
Overall Average AUROC (classification concepts): 0.8316

=== LOGISTIC REGRESSION ===

LR performance (ground truth concepts to activity labels)
Accuracy: 0.3684210526315789
AUROC: 0.8474282354070661

=== FULL CBM STATS x,y,z -> concepts -> activity label ===
Accuracy: 0.5
AUROC: 0.8324172747055812
```

## 2 concepts, periodicity + temporal stability

### CNN Results

```txt
Evaluating model...

=== RESULTS ===

--- Classification Concepts (Discrete) ---
Periodicity - Accuracy: 0.7105, AUROC: 0.8433
Temporal Stability - Accuracy: 0.7368, AUROC: 0.8820

--- Overall Performance ---
Overall Average Accuracy (classification concepts): 72.4%
Overall Average AUROC (classification concepts): 0.8627
```

![alt text](image.png)

### Logistic Regression

```txt
=== Logistic Regression (trained on GT concepts from TRAIN split) ===
Accuracy: 0.3684210526315789
AUROC: 0.8458601350486431
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         6
           1       0.33      0.29      0.31         7
           2       0.67      1.00      0.80         6
           3       0.00      0.00      0.00         6
           4       0.29      0.67      0.40         6
           5       0.25      0.29      0.27         7

    accuracy                           0.37        38
   macro avg       0.26      0.37      0.30        38
weighted avg       0.26      0.37      0.30        38

[[0 0 0 0 5 1]
 [0 2 0 1 1 3]
 [0 0 6 0 0 0]
 [0 0 3 0 3 0]
 [0 0 0 0 4 2]
 [0 4 0 0 1 2]]
```

### Full CBM Results

```txt
=== CNN Results ===

Classification Concepts (Discrete)
Periodicity - Accuracy: 0.7105, AUROC: 0.8426
Temporal Stability - Accuracy: 0.7368, AUROC: 0.9196

Overall Performance
Overall Average Accuracy (classification concepts): 72.4%
Overall Average AUROC (classification concepts): 0.8811

=== LOGISTIC REGRESSION ===

LR performance (ground truth concepts to activity labels)
Accuracy: 0.3684210526315789
AUROC: 0.8458601350486431

=== FULL CBM STATS x,y,z -> concepts -> activity label ===
Accuracy: 0.3684210526315789
AUROC: 0.7841381848438299
```

## 2 concepts, periodicity + coordination

### CNN Results

```txt
Evaluating model...

=== RESULTS ===

--- Classification Concepts (Discrete) ---
Periodicity - Accuracy: 0.7105, AUROC: 0.8280
Coordination - Accuracy: 0.7895, AUROC: 0.8607

--- Overall Performance ---
Overall Average Accuracy (classification concepts): 75.0%
Overall Average AUROC (classification concepts): 0.8444
```

![alt text](image-1.png)

### Logistic Regression

```txt
=== Logistic Regression (trained on GT concepts from TRAIN split) ===
Accuracy: 0.2894736842105263
AUROC: 0.8040574596774194
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         6
           1       0.33      0.29      0.31         7
           2       0.00      0.00      0.00         6
           3       0.33      1.00      0.50         6
           4       0.38      0.50      0.43         6
           5       0.00      0.00      0.00         7

    accuracy                           0.29        38
   macro avg       0.17      0.30      0.21        38
weighted avg       0.17      0.29      0.20        38

[[0 0 0 2 3 1]
 [0 2 0 1 1 3]
 [0 0 0 6 0 0]
 [0 0 0 6 0 0]
 [0 0 0 1 3 2]
 [0 4 0 2 1 0]]
```

### Full CBM Results

```txt
=== CNN Results ===

Classification Concepts (Discrete)
Periodicity - Accuracy: 0.6842, AUROC: 0.8141
Coordination - Accuracy: 0.7368, AUROC: 0.8624

Overall Performance
Overall Average Accuracy (classification concepts): 71.1%
Overall Average AUROC (classification concepts): 0.8383

=== LOGISTIC REGRESSION ===

LR performance (ground truth concepts to activity labels)
Accuracy: 0.2894736842105263
AUROC: 0.8040574596774194

=== FULL CBM STATS x,y,z -> concepts -> activity label ===
Accuracy: 0.3684210526315789
AUROC: 0.7715033762160779
```

## 2 concepts, temporal stability + coordination

### CNN Results

```txt
Evaluating model...

=== RESULTS ===

--- Classification Concepts (Discrete) ---
Temporal Stability - Accuracy: 0.7632, AUROC: 0.8970
Coordination - Accuracy: 0.7105, AUROC: 0.8602

--- Overall Performance ---
Overall Average Accuracy (classification concepts): 73.7%
Overall Average AUROC (classification concepts): 0.8786
```

![alt text](image-2.png)

### Logistic Regression

```txt
=== Logistic Regression (trained on GT concepts from TRAIN split) ===
Accuracy: 0.34210526315789475
AUROC: 0.7528621831797236
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         6
           1       0.20      0.29      0.24         7
           2       0.00      0.00      0.00         6
           3       0.30      0.50      0.38         6
           4       0.60      0.50      0.55         6
           5       0.38      0.71      0.50         7

    accuracy                           0.34        38
   macro avg       0.25      0.33      0.28        38
weighted avg       0.25      0.34      0.28        38

[[0 2 0 0 2 2]
 [0 2 0 1 0 4]
 [0 0 0 6 0 0]
 [0 3 0 3 0 0]
 [0 1 0 0 3 2]
 [0 2 0 0 0 5]]
```

### Full CBM Results

```txt
=== CNN Results ===

Classification Concepts (Discrete)
Temporal Stability - Accuracy: 0.7368, AUROC: 0.8965
Coordination - Accuracy: 0.6842, AUROC: 0.8206

Overall Performance
Overall Average Accuracy (classification concepts): 71.1%
Overall Average AUROC (classification concepts): 0.8586

=== LOGISTIC REGRESSION ===

LR performance (ground truth concepts to activity labels)
Accuracy: 0.34210526315789475
AUROC: 0.7528621831797236

=== FULL CBM STATS x,y,z -> concepts -> activity label ===
Accuracy: 0.2631578947368421
AUROC: 0.741159434203789
```

# WITH RULE BASED - so this is base visual human concepts + rule

## 2 concepts movement_variability + movement_consistency

### CNN Results

```txt
Evaluating model...

=== RESULTS ===

--- Classification Concepts (Discrete) ---
Periodicity - Accuracy: 0.7105, AUROC: 0.8621
Temporal Stability - Accuracy: 0.7368, AUROC: 0.8666
Coordination - Accuracy: 0.6842, AUROC: 0.7655
Movement Variability - Accuracy: 0.9474, AUROC: 0.9918
Movement Consistency - Accuracy: 0.6842, AUROC: 0.8232

--- Overall Performance ---
Overall Average Accuracy (classification concepts): 75.3%
Overall Average AUROC (classification concepts): 0.8618
```

![alt text](image-3.png)

### Logistic Regression

```txt
=== Logistic Regression (trained on GT concepts from TRAIN split) ===
Accuracy: 0.5789473684210527
AUROC: 0.8958293330773172
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         6
           1       0.75      0.86      0.80         7
           2       0.62      0.83      0.71         6
           3       0.75      0.50      0.60         6
           4       0.30      0.50      0.38         6
           5       0.71      0.71      0.71         7

    accuracy                           0.58        38
   macro avg       0.52      0.57      0.53        38
weighted avg       0.53      0.58      0.55        38

[[0 0 0 0 5 1]
 [0 6 0 0 1 0]
 [0 0 5 1 0 0]
 [0 0 3 3 0 0]
 [1 1 0 0 3 1]
 [0 1 0 0 1 5]]
```

### Full CBM Results

```txt
=== CNN Results ===

Classification Concepts (Discrete)
Periodicity - Accuracy: 0.7368, AUROC: 0.8407
Temporal Stability - Accuracy: 0.7368, AUROC: 0.8957
Coordination - Accuracy: 0.7105, AUROC: 0.7880
Movement Variability - Accuracy: 0.9474, AUROC: 0.9908
Movement Consistency - Accuracy: 0.7105, AUROC: 0.7878

Overall Performance
Overall Average Accuracy (classification concepts): 76.8%
Overall Average AUROC (classification concepts): 0.8606

=== LOGISTIC REGRESSION ===

LR performance (ground truth concepts to activity labels)
Accuracy: 0.5789473684210527
AUROC: 0.8958293330773172

=== FULL CBM STATS x,y,z -> concepts -> activity label ===
Accuracy: 0.5263157894736842
AUROC: 0.8382656490015362
```

## 1 concept movement_variability

### CNN Results

```txt
Evaluating model...

=== RESULTS ===

--- Classification Concepts (Discrete) ---
Periodicity - Accuracy: 0.7368, AUROC: 0.8363
Temporal Stability - Accuracy: 0.7632, AUROC: 0.8961
Coordination - Accuracy: 0.7632, AUROC: 0.8246
Movement Variability - Accuracy: 0.9211, AUROC: 0.9899

--- Overall Performance ---
Overall Average Accuracy (classification concepts): 79.6%
Overall Average AUROC (classification concepts): 0.8867
```

![alt text](image-4.png)

### Logistic Regression

```txt
=== Logistic Regression (trained on GT concepts from TRAIN split) ===
Accuracy: 0.6578947368421053
AUROC: 0.9048099078341013
              precision    recall  f1-score   support

           0       0.67      0.33      0.44         6
           1       0.75      0.86      0.80         7
           2       0.67      1.00      0.80         6
           3       1.00      0.50      0.67         6
           4       0.38      0.50      0.43         6
           5       0.71      0.71      0.71         7

    accuracy                           0.66        38
   macro avg       0.70      0.65      0.64        38
weighted avg       0.70      0.66      0.65        38

[[2 0 0 0 3 1]
 [0 6 0 0 1 0]
 [0 0 6 0 0 0]
 [0 0 3 3 0 0]
 [1 1 0 0 3 1]
 [0 1 0 0 1 5]]
```

### Full CBM Results

```txt
=== CNN Results ===

Classification Concepts (Discrete)
Periodicity - Accuracy: 0.7105, AUROC: 0.8275
Temporal Stability - Accuracy: 0.7632, AUROC: 0.8927
Coordination - Accuracy: 0.7895, AUROC: 0.8358
Movement Variability - Accuracy: 0.9211, AUROC: 0.9908

Overall Performance
Overall Average Accuracy (classification concepts): 79.6%
Overall Average AUROC (classification concepts): 0.8867

=== LOGISTIC REGRESSION ===

LR performance (ground truth concepts to activity labels)
Accuracy: 0.6578947368421053
AUROC: 0.9048099078341013

=== FULL CBM STATS x,y,z -> concepts -> activity label ===
Accuracy: 0.6052631578947368
AUROC: 0.8865127368151562
```

## 1 concept movement_consistency

### CNN Results

```txt
=== RESULTS ===

--- Classification Concepts (Discrete) ---
Periodicity - Accuracy: 0.6842, AUROC: 0.8242
Temporal Stability - Accuracy: 0.7368, AUROC: 0.9038
Coordination - Accuracy: 0.7632, AUROC: 0.8776
Movement Consistency - Accuracy: 0.7368, AUROC: 0.8652

--- Overall Performance ---
Overall Average Accuracy (classification concepts): 73.0%
Overall Average AUROC (classification concepts): 0.8677
```

![alt text](image-5.png)

### Logistic Regression

```txt
=== Logistic Regression (trained on GT concepts from TRAIN split) ===
Accuracy: 0.34210526315789475
AUROC: 0.8479962717613928
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         6
           1       0.38      0.43      0.40         7
           2       0.62      0.83      0.71         6
           3       0.50      0.33      0.40         6
           4       0.27      0.50      0.35         6
           5       0.00      0.00      0.00         7

    accuracy                           0.34        38
   macro avg       0.30      0.35      0.31        38
weighted avg       0.29      0.34      0.31        38

[[0 0 0 0 5 1]
 [0 3 0 0 1 3]
 [0 0 5 1 0 0]
 [0 0 3 2 1 0]
 [1 0 0 0 3 2]
 [0 5 0 1 1 0]]
```

### Full CBM Results

```txt
=== CNN Results ===

Classification Concepts (Discrete)
Periodicity - Accuracy: 0.6579, AUROC: 0.8271
Temporal Stability - Accuracy: 0.7368, AUROC: 0.9082
Coordination - Accuracy: 0.7368, AUROC: 0.8789
Movement Consistency - Accuracy: 0.7368, AUROC: 0.8779

Overall Performance
Overall Average Accuracy (classification concepts): 71.7%
Overall Average AUROC (classification concepts): 0.8730

=== LOGISTIC REGRESSION ===

LR performance (ground truth concepts to activity labels)
Accuracy: 0.34210526315789475
AUROC: 0.8479962717613928

=== FULL CBM STATS x,y,z -> concepts -> activity label ===
Accuracy: 0.5263157894736842
AUROC: 0.8026173675115206
```
