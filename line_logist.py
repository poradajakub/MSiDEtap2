import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
import os


# Tworzenie folderu na wyniki, jeśli go nie ma
output_dir = 'logistic_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # stabilizacja numeryczna
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def softmax_gradient_descent(X, y, lr=0.1, epochs=500, batch_size=32):
    m, n = X.shape
    k = y.shape[1]  # liczba klas
    weights = np.zeros((n, k))
    accuracies = []  # lista na dokładność w trakcie treningu
    costs = []  # lista na koszt w trakcie treningu

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            logits = X_batch @ weights
            y_pred = softmax(logits)
            error = y_pred - y_batch
            grad = X_batch.T @ error / batch_size
            weights -= lr * grad

        # Oblicz dokładność i koszt po każdej epoce
        logits_test = X @ weights
        y_pred_test = softmax(logits_test)
        y_pred_test_class = np.argmax(y_pred_test, axis=1)
        y_true_test_class = np.argmax(y, axis=1)
        accuracy = accuracy_score(y_true_test_class, y_pred_test_class)
        accuracies.append(accuracy)

        cost = cross_entropy(y, y_pred_test)
        costs.append(cost)

    return weights, accuracies, costs


def predict_softmax(X, weights):
    logits = X @ weights
    return np.argmax(softmax(logits), axis=1)


def one_hot_encode(y):
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    return y_encoded, encoder


def load_and_preprocess(data_path, numeric_features, categorical_features, target):
    df = pd.read_csv(data_path)
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    all_features = numeric_features + [c for c in df.columns
                                       if any(c.startswith(cat + '_') for cat in categorical_features)]
    X_df = df[all_features].copy()
    y_raw = df[target].to_numpy()

    X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
        X_df, y_raw, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train_df))
    X_test = scaler.transform(imputer.transform(X_test_df))

    X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    y_train_encoded, encoder = one_hot_encode(y_train_raw)
    y_test_encoded = encoder.transform(y_test_raw.reshape(-1, 1))

    return X_train_bias, X_test_bias, y_train_encoded, y_test_encoded, y_test_raw, encoder


def plot_confusion_matrix(y_true, y_pred, target_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Przewidywania')
    plt.ylabel('Rzeczywiste')
    plt.title('Macierz konfuzji')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}_confusion_matrix.png")
    plt.close()


def main():
    data_path = 'ObesityDataSet.csv'
    numeric_features = ['Age', 'Height','Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
        'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]
    target = 'NObeyesdad'

    X_train, X_test, y_train_onehot, y_test_onehot, y_test_raw, encoder = load_and_preprocess(
        data_path, numeric_features, categorical_features, target)

    # Własna implementacja softmax regresji
    weights, accuracies, costs = softmax_gradient_descent(X_train, y_train_onehot, lr=0.1, epochs=1000, batch_size=64)
    logits = X_test @ weights
    y_pred_probs = softmax(logits)
    y_pred_own = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    # Zapisz wyniki do pliku
    with open(f"{output_dir}/results.txt", "w") as f:
        f.write("== Własna softmax regresja ==\n")
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred_own)}\n")
        f.write(f"Cross-Entropy Loss: {cross_entropy(y_test_onehot, y_pred_probs)}\n")
        f.write(classification_report(y_true, y_pred_own, target_names=encoder.categories_[0]))

    # Wykresy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 1001), accuracies, label="Dokładność", color='blue')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność')
    plt.title('Dokładność w trakcie treningu (Własna implementacja)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_plot.png")
    plt.close()

    # Wykres funkcji kosztu
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 1001), costs, label="Cross-Entropy Loss", color='red')
    plt.xlabel('Epoki')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Cross-Entropy Loss w trakcie treningu (Własna implementacja)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cost_plot.png")
    plt.close()

    # Macierz konfuzji z uporządkowanymi klasami
    class_order = [
        'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
        'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]
    plot_confusion_matrix(y_true, y_pred_own, class_order, "own_model")

    # Porównanie z sklearn LogisticRegression
    print("\n== sklearn LogisticRegression ==")
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, np.argmax(y_train_onehot, axis=1))
    y_pred_sklearn = clf.predict(X_test)
    y_proba_sklearn = clf.predict_proba(X_test)

    # Zapisz wyniki dla sklearn do pliku
    with open(f"{output_dir}/results.txt", "a") as f:
        f.write("\n== sklearn LogisticRegression ==\n")
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred_sklearn)}\n")
        f.write(f"Cross-Entropy Loss (log_loss): {log_loss(y_test_onehot, y_proba_sklearn)}\n")
        f.write(classification_report(y_true, y_pred_sklearn, target_names=encoder.categories_[0]))

    # Macierz konfuzji dla sklearn
    plot_confusion_matrix(y_true, y_pred_sklearn, class_order, "sklearn_model")


if __name__ == '__main__':
    main()
