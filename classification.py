from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from utils import CLASS_NAMES, RANDOM_STATE


CLASSIFIER_CONFIGS = {
    'SVM': lambda: SVC(kernel='rbf', C=1.0, gamma='scale',
                       random_state=RANDOM_STATE),
    'SGD': lambda: SGDClassifier(loss='hinge', alpha=0.01, max_iter=1000,
                                 tol=1e-3, random_state=RANDOM_STATE),
    'RandomForest': lambda: RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    'NaiveBayes': lambda: GaussianNB(),
}


def get_classifier(name):
    if name not in CLASSIFIER_CONFIGS:
        raise ValueError(f"Unknown classifier: {name}. "
                         f"Choose from {list(CLASSIFIER_CONFIGS.keys())}")
    return CLASSIFIER_CONFIGS[name]()


def get_all_classifier_names():
    return list(CLASSIFIER_CONFIGS.keys())


def train_classifier(name, X_train, y_train):
    model = get_classifier(name)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted',
                                     zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted',
                               zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted',
                       zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
        ),
        'y_pred': y_pred,
    }
    return results


def train_and_evaluate_all(X_train, y_train, X_test, y_test, verbose=True):
    models = {}
    results = {}

    for name in get_all_classifier_names():
        if verbose:
            print(f"\n  Training {name}...")
        model = train_classifier(name, X_train, y_train)
        res = evaluate_classifier(model, X_test, y_test)
        models[name] = model
        results[name] = res
        if verbose:
            print(f"    Accuracy: {res['accuracy']:.4f}  F1: {res['f1']:.4f}")

    return models, results