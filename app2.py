import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import plotly.graph_objects as go
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

app = dash.Dash(__name__)

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train different models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Layout of the app
app.layout = html.Div([
    html.H1("Machine Learning Model Evaluation Dashboard"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': model_name, 'value': model_name} for model_name in models.keys()],
        value=list(models.keys())[0]
    ),
    html.Div(id='confusion-matrix'),
    html.Div(id='precision-recall-f1'),
    dcc.Graph(id='roc-curve')
])

# Callback to update the display based on model selection
@app.callback(
    [Output('confusion-matrix', 'children'),
     Output('precision-recall-f1', 'children'),
     Output('roc-curve', 'figure')],
    [Input('model-dropdown', 'value')]
)
def update_metrics(selected_model):
    model = models[selected_model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_fig = go.Figure(data=go.Heatmap(z=cm, colorscale='Viridis'))
    confusion_matrix_fig.update_layout(title=f'Confusion Matrix for {selected_model}')

    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision_recall_f1_text = f"Precision: {precision:.3f}<br>Recall: {recall:.3f}<br>F1 Score: {f1:.3f}"

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    roc_curve_fig = go.Figure()
    roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{selected_model} (AUROC = {roc_auc:.3f})'))
    roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
    roc_curve_fig.update_layout(title=f'ROC Curve for {selected_model}', xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate')

    return confusion_matrix_fig, precision_recall_f1_text, roc_curve_fig

if __name__ == '__main__':
    app.run_server(debug=True)
