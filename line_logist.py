import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
import os

output_dir = 'logistic_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def softmax_gradient_descent(X, y, lr=0.1, epochs=500, batch_size=32):
    m, n = X.shape
    k = y.shape[1]
    weights = np.zeros((n, k))
    accuracies = []
    costs = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            logits = X_batch @ weights
            y_pred = softmax(logits)
            error = y_pred - y_batch
            grad = X_batch.T @ error / batch_size
            weights -= lr * grad

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

    return X_df, y_raw


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


def cross_validate_softmax(X_df, y_raw, n_splits=3, lr=0.1, epochs=1000, batch_size=64):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_losses = []
    all_y_true = []
    all_y_pred = []

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False)

    y_encoded_full = encoder.fit_transform(y_raw.reshape(-1, 1))
    target_names = encoder.categories_[0]

    for fold, (train_index, test_index) in enumerate(skf.split(X_df, y_raw)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        X_train_df, X_test_df = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train_raw, y_test_raw = y_raw[train_index], y_raw[test_index]

        X_train_imputed = imputer.fit_transform(X_train_df)
        X_test_imputed = imputer.transform(X_test_df)

        X_train = scaler.fit_transform(X_train_imputed)
        X_test = scaler.transform(X_test_imputed)

        X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        y_train_encoded = encoder.transform(y_train_raw.reshape(-1, 1))
        y_test_encoded = encoder.transform(y_test_raw.reshape(-1, 1))

        weights, _, _ = softmax_gradient_descent(X_train_bias, y_train_encoded, lr=lr, epochs=epochs,
                                                 batch_size=batch_size)

        logits_test = X_test_bias @ weights
        y_pred_probs = softmax(logits_test)
        y_pred_own = np.argmax(y_pred_probs, axis=1)
        y_true_fold = np.argmax(y_test_encoded, axis=1)

        accuracy = accuracy_score(y_true_fold, y_pred_own)
        loss = cross_entropy(y_test_encoded, y_pred_probs)

        fold_accuracies.append(accuracy)
        fold_losses.append(loss)

        all_y_true.extend(y_true_fold)
        all_y_pred.extend(y_pred_own)

        print(f"Accuracy for Fold {fold + 1}: {accuracy:.4f}")
        print(f"Cross-Entropy Loss for Fold {fold + 1}: {loss:.4f}")
        print(classification_report(y_true_fold, y_pred_own, target_names=target_names))

    avg_accuracy = np.mean(fold_accuracies)
    avg_loss = np.mean(fold_losses)

    print(f"\n--- Wyniki walidacji krzyżowej (Własna implementacja) ---")
    print(f"Średnia dokładność (Accuracy) across folds: {avg_accuracy:.4f}")
    print(f"Odchylenie standardowe dokładności: {np.std(fold_accuracies):.4f}")
    print(f"Średni koszt (Cross-Entropy Loss) across folds: {avg_loss:.4f}")
    print(f"Odchylenie standardowe kosztu: {np.std(fold_losses):.4f}")

    plot_confusion_matrix(all_y_true, all_y_pred, target_names, "cross_validation_own_model")

    return fold_accuracies, fold_losses, target_names


def main():
    data_path = 'ObesityDataSet.csv'
    numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
        'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]
    target = 'NObeyesdad'

    X_df, y_raw = load_and_preprocess(data_path, numeric_features, categorical_features, target)

    fold_accuracies_own, fold_losses_own, target_names = cross_validate_softmax(X_df, y_raw, n_splits=3, lr=0.1,
                                                                                epochs=1000, batch_size=64)

    print("\n--- Porównanie z sklearn LogisticRegression (Walidacja Krzyżowa) ---")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)
    sklearn_fold_accuracies = []
    sklearn_fold_losses = []

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    for fold, (train_index, test_index) in enumerate(skf.split(X_df, y_raw)):
        print(f"\n--- Sklearn Fold {fold + 1}/3 ---")
        X_train_df, X_test_df = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train_raw, y_test_raw = y_raw[train_index], y_raw[test_index]

        X_train_imputed = imputer.fit_transform(X_train_df)
        X_test_imputed = imputer.transform(X_test_df)

        X_train = scaler.fit_transform(X_train_imputed)
        X_test = scaler.transform(X_test_imputed)

        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(X_train, y_train_raw)
        y_pred_sklearn = clf.predict(X_test)
        y_proba_sklearn = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test_raw, y_pred_sklearn)
        loss = log_loss(y_test_raw,
                        y_proba_sklearn)

        sklearn_fold_accuracies.append(accuracy)
        sklearn_fold_losses.append(
            loss)

        print(f"Sklearn Accuracy for Fold {fold + 1}: {accuracy:.4f}")
        print(f"Sklearn Cross-Entropy Loss for Fold {fold + 1}: {loss:.4f}")
        print(classification_report(y_test_raw, y_pred_sklearn, target_names=target_names))

    print(f"\n--- Wyniki walidacji krzyżowej (sklearn LogisticRegression) ---")
    print(f"Średnia dokładność (Accuracy) across folds: {np.mean(sklearn_fold_accuracies):.4f}")
    print(f"Odchylenie standardowe dokładności: {np.std(sklearn_fold_accuracies):.4f}")
    print(f"Średni koszt (Cross-Entropy Loss) across folds: {np.mean(sklearn_fold_losses):.4f}")
    print(f"Odchylenie standardowe kosztu: {np.std(sklearn_fold_losses):.4f}")


if __name__ == '__main__':
    main()
