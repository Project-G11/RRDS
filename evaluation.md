# Dialog System Evaluation

## 1. Quantitative Evaluation

### 1.1 An Explanation of the Quantitative Evaluation Metrics

### 1.2 Baseline Systems

### 1.3 Decision Tree

Decision Tree classifier accuracy: 0.97909

Classification Report:

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| ack       | 0.75      | 0.60   | 0.67     | 5       |
| affirm    | 0.99      | 0.99   | 0.99     | 180     |
| bye       | 0.97      | 1.00   | 0.99     | 35      |
| confirm   | 0.78      | 0.82   | 0.80     | 22      |
| deny      | 0.86      | 1.00   | 0.92     | 6       |
| hello     | 1.00      | 1.00   | 1.00     | 14      |
| inform    | 0.99      | 0.97   | 0.98     | 1532    |
| negate    | 1.00      | 1.00   | 1.00     | 69      |
| null      | 0.86      | 0.98   | 0.92     | 232     |
| repeat    | 1.00      | 0.67   | 0.80     | 3       |
| reqalts   | 0.97      | 0.97   | 0.97     | 279     |
| reqmore   | 1.00      | 1.00   | 1.00     | 1       |
| request   | 0.99      | 1.00   | 0.99     | 972     |
| restart   | 1.00      | 0.50   | 0.67     | 2       |
| thankyou  | 1.00      | 1.00   | 1.00     | 474     |
|-----------|-----------|--------|----------|---------|
| Accuracy  |           |        | 0.98     | 3826    |
| Macro Avg | 0.94      | 0.90   | 0.91     | 3826    |
| Weighted Avg | 0.98   | 0.98   | 0.98     | 3826    |


### 1.4 Logistic Regression

Accuracy: 0.98

Classification Report:

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| ack       | 0.00      | 0.00   | 0.00     | 5       |
| affirm    | 0.99      | 0.98   | 0.99     | 180     |
| bye       | 0.94      | 0.91   | 0.93     | 35      |
| confirm   | 0.81      | 0.77   | 0.79     | 22      |
| deny      | 1.00      | 0.50   | 0.67     | 6       |
| hello     | 1.00      | 0.93   | 0.96     | 14      |
| inform    | 0.98      | 0.99   | 0.98     | 1532    |
| negate    | 1.00      | 0.99   | 0.99     | 69      |
| null      | 0.97      | 0.93   | 0.95     | 232     |
| repeat    | 1.00      | 0.67   | 0.80     | 3       |
| reqalts   | 0.95      | 0.98   | 0.97     | 279     |
| reqmore   | 0.00      | 0.00   | 0.00     | 1       |
| request   | 1.00      | 0.99   | 1.00     | 972     |
| restart   | 1.00      | 1.00   | 1.00     | 2       |
| thankyou  | 0.99      | 1.00   | 0.99     | 474     |
|-----------|-----------|--------|----------|---------|
| Accuracy  |           |        | 0.98     | 3826    |
| Macro Avg | 0.84      | 0.78   | 0.80     | 3826    |
| Weighted Avg | 0.98   | 0.98   | 0.98     | 3826    |




### 1.5 Feed-Forward Neural Network 

Feedforward Neural Network classifier accuracy: 0.986147403717041

Feedforward Neural Network classifier loss: 0.05195702612400055

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| ack       | 0.00      | 0.00   | 0.00     | 5       |
| affirm    | 0.99      | 0.98   | 0.99     | 180     |
| bye       | 0.94      | 0.97   | 0.96     | 35      |
| confirm   | 0.87      | 0.91   | 0.89     | 22      |
| deny      | 1.00      | 0.50   | 0.67     | 6       |
| hello     | 1.00      | 0.86   | 0.92     | 14      |
| inform    | 0.99      | 0.99   | 0.99     | 1532    |
| negate    | 1.00      | 1.00   | 1.00     | 69      |
| null      | 0.97      | 0.94   | 0.95     | 232     |
| repeat    | 1.00      | 0.67   | 0.80     | 3       |
| reqalts   | 0.97      | 0.96   | 0.97     | 279     |
| reqmore   | 0.00      | 0.00   | 0.00     | 1       |
| request   | 1.00      | 1.00   | 1.00     | 972     |
| restart   | 0.00      | 0.00   | 0.00     | 2       |
| thankyou  | 1.00      | 0.99   | 1.00     | 474     |
|-----------|-----------|--------|----------|---------|
| Micro Avg | 0.99      | 0.98   | 0.99     | 3826    |
| Macro Avg | 0.78      | 0.72   | 0.74     | 3826    |
| Weighted Avg | 0.99   | 0.98   | 0.98     | 3826    |

## 2. Error Analysis

## 3. Difficult Cases

## 4. System Comparison

