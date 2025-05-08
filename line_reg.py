import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Funkcja do regresji analitycznej (closed-form)
def closed_form_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    return XtX_inv @ Xty

# Funkcja do wczytywania i przetwarzania danych
def load_and_preprocess(
        data_path: str,
        numeric_features: list[str],
        categorical_features: list[str],
        target: str,
        test_size: float = 0.2,
        random_state: int = 21
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(data_path)
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    all_features = numeric_features + [c for c in df.columns
                                       if any(c.startswith(cat + '_') for cat in categorical_features)]

    X_df = df[all_features].copy()
    y = df[target].to_numpy()

    # Podział danych na treningowe i testowe
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state
    )

    # Imputacja brakujących danych oraz skalowanie
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_df))
    X_test = scaler.transform(imputer.transform(X_test_df))

    # Dodanie biasu (jedynki) do danych
    X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    return X_train_bias, X_test_bias, y_train, y_test, all_features

# Funkcja obliczająca metryki regresji
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Funkcja do zapisu wyników do pliku
def save_results(
        results_dir: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    # Zapis danych do pliku CSV
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).to_csv(
        os.path.join(results_dir, 'predictions_closed_form.csv'), index=False
    )
    pd.Series(weights.flatten(), name='weight').to_csv(
        os.path.join(results_dir, 'weights_closed_form.csv'), index=False
    )

    # Wykres: Porównanie rzeczywistych i przewidywanych wartości
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
    plt.xlabel('Rzeczywiste wartości')
    plt.ylabel('Przewidywane wartości')
    plt.title('Rzeczywiste vs Przewidywane')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'true_vs_pred.png'))
    plt.close()

    # Histogram residuów
    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Residua')
    plt.ylabel('Liczba próbek')
    plt.title('Histogram residuów')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'residuals_hist.png'))
    plt.close()

    # Residua vs wartości predykowane
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Przewidywane wartości')
    plt.ylabel('Residua')
    plt.title('Residua vs Przewidywane')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'residuals_vs_pred.png'))
    plt.close()

# Główna funkcja
def main():
    data_path = 'ObesityDataSet.csv'
    numeric_features = ['Age', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
        'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'
    ]
    target = 'Weight'
    results_dir = 'regression_results'

    # Wczytanie i przetwarzanie danych
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        data_path, numeric_features, categorical_features, target
    )

    # Wyznaczenie wag za pomocą regresji analitycznej
    weights = closed_form_regression(X_train, y_train)

    # Predykcja na danych testowych
    y_pred = X_test @ weights

    # Obliczenie metryk modelu
    metrics = regression_metrics(y_test, y_pred)
    print("Metryki modelu closed-form:")
    for name, val in metrics.items():
        print(f" - {name}: {val:.4f}")

    # Zapis wyników do pliku
    save_results(results_dir, y_test, y_pred, weights)
    print(f"Wyniki zapisane w katalogu '{results_dir}'.")

    # Porównanie z modelem sklearn
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_sklearn = lr.predict(X_test)
    sk_metrics = regression_metrics(y_test, y_pred_sklearn)
    print("\nMetryki modelu sklearn LinearRegression:")
    for name, val in sk_metrics.items():
        print(f" - {name}: {val:.4f}")

if __name__ == '__main__':
    main()
