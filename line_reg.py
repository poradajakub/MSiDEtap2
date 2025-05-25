import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def closed_form_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    return XtX_inv @ Xty


def load_and_preprocess_for_cv(
        data_path: str,
        numeric_features: list[str],
        categorical_features: list[str],
        target: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(data_path)
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    all_features = numeric_features + [c for c in df.columns
                                       if any(c.startswith(cat + '_') for cat in categorical_features)]

    X_df = df[all_features].copy()
    y = df[target].to_numpy()

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_df)

    X_bias = np.hstack([np.ones((X_imputed.shape[0], 1)), X_imputed])

    return X_bias, y, all_features


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        r2 = 0 if ss_res == 0 else -np.inf
    else:
        r2 = 1 - ss_res / ss_tot
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


def save_results(
        results_dir: str,
        fold: int,  # Add fold number
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
) -> None:
    fold_dir = os.path.join(results_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).to_csv(
        os.path.join(fold_dir, 'predictions_closed_form.csv'), index=False
    )
    pd.Series(weights.flatten(), name='weight').to_csv(
        os.path.join(fold_dir, 'weights_closed_form.csv'), index=False
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
    plt.xlabel('Rzeczywiste wartości')
    plt.ylabel('Przewidywane wartości')
    plt.title(f'Fold {fold}: Rzeczywiste vs Przewidywane')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'true_vs_pred.png'))
    plt.close()

    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Residua')
    plt.ylabel('Liczba próbek')
    plt.title(f'Fold {fold}: Histogram residuów')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'residuals_hist.png'))
    plt.close()

    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Przewidywane wartości')
    plt.ylabel('Residua')
    plt.title(f'Fold {fold}: Residua vs Przewidywane')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'residuals_vs_pred.png'))
    plt.close()


def main_cv():
    data_path = 'ObesityDataSet.csv'
    numeric_features = ['Age', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
        'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]
    target = 'Weight'
    results_dir = 'regression_results_cv'
    n_splits = 3

    X_data, y_data, feature_names_with_bias = load_and_preprocess_for_cv(
        data_path, numeric_features, categorical_features, target
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=21)

    fold_metrics = []
    all_fold_results_df = pd.DataFrame()

    print(f"Rozpoczynanie {n_splits}-krotnej walidacji krzyżowej...\n")

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X_data)):
        print(f"--- Fold {fold_idx + 1}/{n_splits} ---")

        X_train_fold_unscaled, X_val_fold_unscaled = X_data[train_index], X_data[val_index]
        y_train_fold, y_val_fold = y_data[train_index], y_data[val_index]

        scaler = StandardScaler()
        X_train_fold_scaled_features = scaler.fit_transform(X_train_fold_unscaled[:, 1:])
        X_train_fold = np.hstack(
            [X_train_fold_unscaled[:, [0]], X_train_fold_scaled_features])

        X_val_fold_scaled_features = scaler.transform(X_val_fold_unscaled[:, 1:])
        X_val_fold = np.hstack([X_val_fold_unscaled[:, [0]], X_val_fold_scaled_features])

        weights = closed_form_regression(X_train_fold, y_train_fold)

        y_pred_fold = X_val_fold @ weights

        metrics = regression_metrics(y_val_fold, y_pred_fold)
        fold_metrics.append(metrics)
        print(f"Metryki dla Foldu {fold_idx + 1}:")
        for name, val in metrics.items():
            print(f" - {name}: {val:.4f}")
        print("-" * 20)

        # Zapis wyników
        # save_results(results_dir, fold_idx + 1, y_val_fold, y_pred_fold, weights)

        fold_results = pd.DataFrame([metrics])
        fold_results['Fold'] = fold_idx + 1
        all_fold_results_df = pd.concat([all_fold_results_df, fold_results], ignore_index=True)

    print("\n--- Podsumowanie walidacji krzyżowej ---")

    all_fold_results_df = all_fold_results_df[
        ['Fold'] + [col for col in all_fold_results_df.columns if col != 'Fold']]
    print(all_fold_results_df.to_string(index=False))

    mean_metrics = {metric_name: np.mean([m[metric_name] for m in fold_metrics]) for metric_name in fold_metrics[0]}
    std_metrics = {metric_name: np.std([m[metric_name] for m in fold_metrics]) for metric_name in fold_metrics[0]}

    print("\nŚrednie metryki ze wszystkich foldów:")
    for name, val in mean_metrics.items():
        print(f" - Średni {name}: {val:.4f} (std: {std_metrics[name]:.4f})")

    print("\nAnaliza podobieństwa wyników na podzbiorach:")
    r2_scores = [m['R2'] for m in fold_metrics]
    if len(set(f"{score:.2f}" for score in r2_scores)) == 1:
        print("Wyniki R2 są bardzo podobne na wszystkich podzbiorach.")
        print(
            "Oznacza to, że model generalizuje w spójny sposób na różnych częściach danych, a wydajność nie jest silnie zależna od konkretnego podziału na zbiór treningowy i walidacyjny.")
    else:
        print("Wyniki R2 różnią się znacząco między podzbiorami.")
        print("Może to oznaczać, że:")
        print("  - Model jest niestabilny: Niewielkie zmiany w danych treningowych prowadzą do różnych wyników.")
        print(
            "  - Dane są niejednorodne: Niektóre foldy mogą zawierać próbki, które są trudniejsze do przewidzenia lub różnią się charakterystyką od innych.")
        print(
            "  - Zbiór danych jest mały: Każdy fold reprezentuje znaczną część danych, a różnice między nimi mogą być bardziej wyraźne.")
        print(
            "  - Może występować problem z overfittingiem na niektórych foldach, jeśli model jest zbyt złożony dla danej porcji danych.")
        print("Wartości R2 dla poszczególnych foldów:", [f"{score:.4f}" for score in r2_scores])


if __name__ == '__main__':
    # Funkcja main dla etapu 3
    main_cv()
    # Funkcja main dla etapu 2
    # main()
