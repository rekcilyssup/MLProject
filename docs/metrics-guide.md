# Metrics Guide

## Accuracy
Overall correct predictions. Can be misleading when classes are imbalanced.

## ROC-AUC
Measures ranking quality across thresholds. Higher is better.

## Average Precision (AP)
Area under precision-recall curve. Better reflects minority-class performance.

## F1 Score
Harmonic mean of precision and recall at a chosen threshold.

## Precision and Recall
Precision measures how many predicted churns are correct.
Recall measures how many actual churns are detected.

## Threshold Tuning
We compute the best F1 threshold from the PR curve and report it.
For real use, optimize based on business costs.
