# Dialog System Evaluation

## 1. Quantitative Evaluation

### 1.1 An Explanation of the Quantitative Evaluation Metrics

In order to examine the three different classification models
a common set of metrics had to be decided on. The most common
metric used to judge all models' performance is the overall
accuracy. It is the first value presented for all systems,
including the baseline models. 

Additionally, a classification report was carried out on all 
the models individually, comparing their ability to 
correctly classify the test cases into their corresponding 
categories. Depending on the systems' performance an
additional four metrics were derived: Precision, measuring
the accuracy of positive predictions; Recall, measuring
a model's ability to identify all relevant instances of a
class; Support, which maintains the number of instances of 
a class in a data set. Weighted accuracy was also derived. 

Overall, all these metrics combined give a good measure of 
the systems' performance in their classification tasks. 

### 1.2 Baseline Systems

Majority class model accuracy is 0.40041819132253004

Keyword matching model accuracy is 0.8112911657083115

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

Classification report:

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
| Accuracy  | 0.99      | 0.98   | 0.99     | 3826    |
| Macro Avg | 0.78      | 0.72   | 0.74     | 3826    |
| Weighted Avg | 0.99   | 0.98   | 0.98     | 3826    |

### 1.6 Overall Comparison

Overall: 

| Metric               | Decision Tree | Logistic Regression | Feedforward Neural Network |
|----------------------|---------------|----------------------|-----------------------------|
| Accuracy             | 0.98          | 0.98                 | 0.986                       |
| Macro Avg Precision  | 0.94          | 0.84                 | 0.78                        |
| Macro Avg Recall     | 0.90          | 0.78                 | 0.72                        |
| Macro Avg F1-Score   | 0.91          | 0.80                 | 0.74                        |
| Weighted Avg Precision | 0.98       | 0.98                 | 0.99                        |
| Weighted Avg Recall  | 0.98          | 0.98                 | 0.98                        |
| Weighted Avg F1-Score| 0.98          | 0.98                 | 0.98                        |


As can be seen from the above table the three models all have desirable levels of accuracy, around 98% for all cases. 

The Decision Tree model has high precision, recall, and F1-scores for most classes, with strong macro and weighted average metrics.

The Logistic Regression model also exhibits high precision and recall, but some classes have lower performance metrics.

The Feedforward Neural Network (FNN) performs well, with high accuracy, but it has lower precision, recall, and F1-scores for some classes. 

It can be concluded from these findings that all the models have good measurable classification rates and are quite 
satisfactory for the performance of restaurant recommendation duties. 


## 2. Error Analysis

To identify which specific dialog acts (categories) each model had the most trouble with individually, we can look at the F1-scores for each category and select those with the lowest F1-scores. Additionally, we can also consider classes with lower precision and recall values.

### 2.1 Decision Tree Model:

The "repeat" class has an F1-score of 0.80, which is relatively lower than other classes.

The "reqmore" class has an F1-score of 0.00, indicating poor performance for this class.

### 2.2 Logistic Regression Model:

The "ack" class has an F1-score of 0.00, indicating that the model struggled to predict this class.

The "reqmore" class has an F1-score of 0.00, indicating a lack of understanding.

### 2.3 Feedforward Neural Network (FNN) Model:

Similar to the Decision Tree and Logistic Regression models, the "ack" class has an F1-score of 0.00, indicating difficulties in predicting this class.

The "repeat" class has an F1-score of 0.80, which is relatively lower than other classes.

### 2.4 Considering the models collectively:

Altogether, the "ack" and "reqmore" classes are consistently challenging for all models. These classes had low F1-scores in all three models, meaning they are difficult for the models to classify universally.

### 2.5 Explanation of the results

Of the possible factors that can be considered to have caused the observed classes to have such low results 
here are a few of the more likely. 

Data imbalance is the most likely factor that contributes toward the low prediction results. The classes that the models
were found to be performing poorly all universally feature a lack of samples for the models to train on. 

It is additionally possible that in addition to the low amount samples, the low-performing classes' samples also 
fail to accurately capture the features of those classes.

Finally, it is possible that due to the wide amount of human responses that can be used to acknowledge or indicate
a need for additional data the model is simply not able to generalize enough to satisfy the test. 


## 3. Difficult Cases

Following the instructions suggested in the evaluation task, two types of difficult cases were presented to the three 
models. Firstly, a gramatically incorrect statement, and secondly, a negation statement.

For the non-fluent statements none of the models was able to correctly discern what the user was trying to imply.
The models, being unequipped with either such a wide pool of mistyped class instances or a routine to discern 
typos, fell back to simply informing the user of the system to retype their request.

And as for the negation statement, its results are a bit more varied. In all the three stages of restaurant specification, the decision tree and logistic regression tree failed to determine that the user intended the food to NOT be expensive, for example, or that the restaurant ought NOT to be in south of town - which is sensible as the underlying algorithms was not equipped to deal with this sort of response. 

Only in the case of the neural network was the system capable of determining when a user wants a restaurant to NOT be cheap. For the other two factors, category and location, the neural network performed as "poorly" as the others. 

## 4. System Comparison

