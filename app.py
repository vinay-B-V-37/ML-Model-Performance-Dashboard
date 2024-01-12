from flask import Flask, jsonify
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64  
from PIL import Image 

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def encode_image_to_base64(image):
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close()

    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    return img_base64

def generate_roc_curve_plot(X_test, y_test, model):
    plt.figure(figsize=(8, 8))
    
    model_probs = model.predict_proba(X_test)[:, 1]
    model_auc = roc_auc_score(y_test, model_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)

    plt.plot(
        model_fpr,
        model_tpr,
        marker='.',
        label=f'{model.__class__.__name__} (AUROC = {model_auc:.3f})'
    )

    plt.title('ROC Plot')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    img_base64 = encode_image_to_base64(plt)

    return f'{img_base64}'

def generate_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    print(ConfusionMatrixDisplay(confusion_matrix=cm).plot())
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    img_base64 = encode_image_to_base64(plt)

    return f'{img_base64}'

def generate_metrics(model, X_test, y_test):

    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'test route'})

@app.route('/models', methods=['GET'])
def get_models_info():
    X, y = make_classification(n_samples=2000, n_classes=2, n_features=10, random_state=0)
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    svm = SVC(probability=True, random_state=0)
    svm.fit(X_train, y_train)

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(max_features=5, n_estimators=500, random_state=0)
    rf.fit(X_train, y_train)

    models = {'SVM': svm, 'Decision Tree': dt, 'Random Forest': rf}
    
    models_info = {}
    for model_name, model_instance in models.items():
        roc_plot_url = generate_roc_curve_plot(X_test, y_test, model_instance)
        cm_url = generate_confusion_matrix(model_instance, X_test, y_test)
        metrics = generate_metrics(model_instance, X_test, y_test)

        models_info[model_name] = {
            'roc_plot_url': roc_plot_url,
            'confusion_matrix_url': cm_url,
            'metrics': metrics
        }

    return jsonify(models_info)

if __name__ == '__main__':
    app.run(debug=True)
