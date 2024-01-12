# ML-Model-Performance-Dashboard
# Machine Learning Model Evaluation with Flask

## Introduction

This Flask application aims to address the need for evaluating the accuracy of various machine learning algorithms. The primary focus is on three popular classifiers: Support Vector Machine (SVM), Decision Tree, and Random Forest. The code provides endpoints to assess the performance of these models, including generating ROC plots, confusion matrices, and key metrics such as precision, recall, and F1 score.

## Use Case and Problem Statement

In many machine learning projects, it is crucial to evaluate and compare different models to choose the most suitable one for a given task. This application simplifies the process of model evaluation by offering a web interface to visualize key performance metrics. Users can quickly assess the effectiveness of different classifiers in solving binary classification problems.

## Functionality

The Flask application exposes two main endpoints:

1. `/test`: A simple test route returning a JSON message. This endpoint is useful for checking the server's functionality.

2. `/models`: This endpoint generates and evaluates three machine learning models - SVM, Decision Tree, and Random Forest. It provides information on each model, including ROC plots, confusion matrices, and key metrics.

## How to Use

1. Ensure you have Flask and other required libraries installed. You can install them using:

   ```bash
   pip install Flask scikit-learn matplotlib pillow flask-cors
   ```

2. Run the Flask application:

   ```bash
   python <filename>.py
   ```

   Replace `<filename>` with the actual name of your Python script.

3. Access the endpoints using a web browser or a tool like `curl` or `Postman`.

## Accuracy Measurements

### ROC Plot

The ROC (Receiver Operating Characteristic) plot visually displays the trade-off between true positive rate and false positive rate. The area under the ROC curve (AUROC) provides a measure of the model's discrimination ability.

### Confusion Matrix

The confusion matrix summarizes the performance of a classification algorithm. It shows the number of true positives, true negatives, false positives, and false negatives, allowing for a detailed understanding of the model's performance.

### Metrics

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to the all observations in the actual class.
- **F1 Score**: The weighted average of precision and recall, providing a balance between the two metrics.

## Conclusion

This Flask application serves as a valuable tool for evaluating the performance of machine learning models. Users can easily visualize and compare the accuracy of SVM, Decision Tree, and Random Forest classifiers, aiding in the selection of the most suitable model for a given task.
