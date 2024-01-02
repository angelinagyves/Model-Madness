# Comparing 10 different machine learning models to find the best one for breast cancer classification

## To replicate:
1. Download .ipynb file
2. Upload to Jupyter Notebook or Google Colab
3. Run!

## Logistic Regression

Logistic Regression is a machine learning model that utilizes binary classification to categorize numerical data.

Results from the notebook:

    Model: Logistic Regression
    Confusion Matrix:
    [[ 62   1]
     [  2 106]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.97      0.98      0.98        63
               1       0.99      0.98      0.99       108
    
        accuracy                           0.98       171
       macro avg       0.98      0.98      0.98       171
    weighted avg       0.98      0.98      0.98       171
    
    AUC Score: 0.9980893592004703

Logistic regression had a precision of 0.97, a recall of 0.98, and an F1-Score of 0.98.

## K-Nearest Neighbors

K-Nearest Neighbors is a supervised classification algorithm that makes predictions based on the majority class of its k-nearest neighbors.

Results from the notebook:

    Model: K-Nearest Neighbors
    Confusion Matrix:
    [[ 59   4]
    [  3 105]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.95      0.94      0.94        63
               1       0.96      0.97      0.97       108
    
        accuracy                           0.96       171
       macro avg       0.96      0.95      0.96       171
    weighted avg       0.96      0.96      0.96       171
    
    AUC Score: 0.9776601998824221

K-Nearest Negihbors had a precision of 0.95, a recall of 0.94, and an F1-Score of 0.94.

## Support Vector Machine(SVM)

SVMs are a binary multiclassification algorithm that finds a hyperplane the maximizes the margin between classes. It handles both linear and non-linear classification through kernels.

Results from the notebook: 

    Model: Support Vector Machine
    Confusion Matrix:
    [[ 61   2]
     [  3 105]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.95      0.97      0.96        63
               1       0.98      0.97      0.98       108
    
        accuracy                           0.97       171
       macro avg       0.97      0.97      0.97       171
    weighted avg       0.97      0.97      0.97       171
    
    AUC Score: 0.9964726631393297

The SVM had a precision of 0.95, a recall of 0.97, and an F1-Score of 0.96.

## Decision Tree Classifier

Decision Tree Classifiers utilize a tree-like structure with nodes (feature tests) and branches (possible outcomes) to classify and for regression.

Results from notebook:

    Model: Support Vector Machine
    Confusion Matrix:
    [[ 61   2]
     [  3 105]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.95      0.97      0.96        63
               1       0.98      0.97      0.98       108
    
        accuracy                           0.97       171
       macro avg       0.97      0.97      0.97       171
    weighted avg       0.97      0.97      0.97       171
    
    AUC Score: 0.9964726631393297

The decision tree classifier had a precision of 0.88, a recall of 0.93, and am F1-Score of 0.92.

## Random Forest Classifier 

Random Forest Classifiers is an ensemble learning method that combines the predictions of multiple decision trees. It reduces the overfitting and lack of model stability that is often present with decision trees.

Results from the notebook:

    Model: Random Forest
    Confusion Matrix:
    [[ 60   3]
     [  2 106]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.97      0.95      0.96        63
               1       0.97      0.98      0.98       108
    
        accuracy                           0.97       171
       macro avg       0.97      0.97      0.97       171
    weighted avg       0.97      0.97      0.97       171
    
    AUC Score: 0.9963256907701352

The random forest classifier had a precision of 0.97, a recall of 0.95, and an F1-Score of 0.96.

## Gradient Boosting Classifier

The Gradient Boosting Classifier is a machine learning model that builds multiple decision trees sequentially in which each sebsequent tree corrects the errors made by its previous tree.

Results from the notebook:

    Model: Gradient Boosting
    Confusion Matrix:
    [[ 59   4]
     [  3 105]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.95      0.94      0.94        63
               1       0.96      0.97      0.97       108
    
        accuracy                           0.96       171
       macro avg       0.96      0.95      0.96       171
    weighted avg       0.96      0.96      0.96       171
    
    AUC Score: 0.9957378012933569

Gradient boosting had a precision of 0.95, a recall of 0.94, and an F1-Score of 0.94.

## Naive Bayes Model

The Naive Bayes model is a probabilistic classfication algorithm based on Baye's theorem that assumes conditional indeoendence of the features of a given class.

Results from the notebook:

    Model: Naive Bayes
    Confusion Matrix:
    [[ 57   6]
     [  5 103]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.92      0.90      0.91        63
               1       0.94      0.95      0.95       108
    
        accuracy                           0.94       171
       macro avg       0.93      0.93      0.93       171
    weighted avg       0.94      0.94      0.94       171
    
    AUC Score: 0.9926513815402704

The Naive Bayes model had a precision of 0.92, a recall of 0.90, and an F1-Score of 0.91.

## Neural Network (MLP)

A neural network is a deep learning model composed of interconnected layers of neurons.

Results from notebook:

    Model: Neural Network (MLP Classifier)
    Confusion Matrix:
    [[ 61   2]
     [  2 106]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.97      0.97      0.97        63
               1       0.98      0.98      0.98       108
    
        accuracy                           0.98       171
       macro avg       0.97      0.97      0.97       171
    weighted avg       0.98      0.98      0.98       171
    
    AUC Score: 0.9966196355085244

The neural network had a precision of 0.97, a recall of 0.97, and an F1-score of 0.97.

## AdaBoost

AdaBoost classifiers are ensemble learning models that combine multiple weak learners to make a strong one (typically shallow decision trees).

Resutls from notebook:

    Model: AdaBoost
    Confusion Matrix:
    [[ 61   2]
     [  2 106]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.97      0.97      0.97        63
               1       0.98      0.98      0.98       108
    
        accuracy                           0.98       171
       macro avg       0.97      0.97      0.97       171
    weighted avg       0.98      0.98      0.98       171
    
    AUC Score: 0.9961787184009406

AdaBoost had a precision of 0.97, recall of 0.97, and an F1 score of 0.97.

## XGBoost

The XGBoost classifier is an optimized gradient boosting library that is especially known for speed, efficiency and performance.

Results from notebook:

    Model: XGBoost
    Confusion Matrix:
    [[ 61   2]
     [  3 105]]
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.95      0.97      0.96        63
               1       0.98      0.97      0.98       108
    
        accuracy                           0.97       171
       macro avg       0.97      0.97      0.97       171
    weighted avg       0.97      0.97      0.97       171
    
    AUC Score: 0.9944150499706055

XGBoost has a precision of 0.95, recall of 0.97, and F1-Score of 0.96.

## Final Rankings for Precision
1. Logistic Regression, Random Forest Classifier, Neural Network, AdaBoost- 0.97

All 4 of these models performed at a precision accuracy of 97%.

## Final Rankings for Recall
1. Logistic Regression- 0.98
2. Support Vector Machine, Neural Network, AdaBoost, XGBoost- 0.97

Logistic Regression performed at a recall accuracy of 98%, and SVMs, MLP, AdaBoost, and XGBoost had an accuracy of 97%.

## Final Rankings for F1-Score
1. Logistic Regression- 0.98
2. Neural Network, AdaBoost- 0.97

Logistic Regression had the best F1-Score predicition with an accuracy of 98%, and MLP and AdaBoost had the second best F1-Score preicition with an accuracy of 97%.

## Conclusion

Overall, taking into account precision, recall, and F1-Score accuracies, logistic regression seems to be the highest performing model for breast cancer classificaiton. The neural network as well as AdaBoost also performed at nearly the same accuracy.
