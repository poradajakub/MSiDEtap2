import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Funkcja do wczytania danych z pliku CSV
def load_data(file_path, log_file):
    if not os.path.isfile(file_path):  # Sprawdzanie, czy plik istnieje
        message = f"Plik nie istnieje: {file_path}\n"
        print(message)
        with open(log_file, 'a') as log:
            log.write(message)
        raise FileNotFoundError(f"Plik nie istnieje: {file_path}")
    df = pd.read_csv(file_path)  # Wczytanie danych do DataFrame
    message = f"Dane wczytane z {file_path}\n"
    print(message)
    with open(log_file, 'a') as log:
        log.write(message)
    return df


# Funkcja do przetwarzania danych - dzielenie na cechy i etykiety
def preprocess_data(df, target_column, log_file):
    # Zastępowanie wartości w kolumnie 'CALC'
    df['CALC'] = df['CALC'].replace('Always', 'Frequently')

    if target_column not in df.columns:  # Sprawdzenie, czy kolumna docelowa istnieje
        message = f"Brak kolumny docelowej: {target_column}\n"
        print(message)
        with open(log_file, 'a') as log:
            log.write(message)
        raise ValueError

    # Podział danych na cechy (X) i etykiety (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Wybór cech numerycznych i kategorycznych
    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Pipeline dla cech numerycznych: imputacja i standaryzacja
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                         ('scaler', StandardScaler())])

    # Pipeline dla cech kategorycznych: imputacja i kodowanie OneHot
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    # ColumnTransformer łączy te pipe'y i aplikuje je do odpowiednich cech
    pre = ColumnTransformer([('num', num_pipe, num_feats),
                             ('cat', cat_pipe, cat_feats)])

    message = "Preprocessor skonfigurowany\n"
    print(message)
    with open(log_file, 'a') as log:
        log.write(message)

    return X, y, pre


# Funkcja do rysowania macierzy pomyłek (confusion matrix)
def plot_confusion_matrix(cm, class_names, results_dir, model_name, log_file):
    fig, ax = plt.subplots(figsize=(8, 6))  # Tworzenie wykresu
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]  # Normalizacja macierzy
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')  # Rysowanie wykresu
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proporcja', rotation=270, labelpad=15)  # Etykieta na pasku kolorów

    ax.set_title(f"Confusion Matrix: {model_name}")  # Tytuł wykresu
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')  # Etykiety dla osi X
    ax.set_yticklabels(class_names)  # Etykiety dla osi Y
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    # Dodanie wartości i procentów do komórek macierzy
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = cm[i, j]
            p = cm_norm[i, j] * 100
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{v}\n{p:.1f}%", ha='center', va='center', color=color)

    plt.tight_layout()  # Optymalizacja układu wykresu
    os.makedirs(results_dir, exist_ok=True)  # Tworzenie katalogu, jeśli nie istnieje
    path = os.path.join(results_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    fig.savefig(path, dpi=150)  # Zapisanie wykresu do pliku
    plt.close(fig)  # Zamknięcie wykresu

    message = f"Zapisano macierz: {path}\n"
    print(message)
    with open(log_file, 'a') as log:
        log.write(message)


# Funkcja do rysowania porównania wyników różnych modeli
def plot_model_comparison(df_metrics, results_dir, log_file):
    os.makedirs(results_dir, exist_ok=True)  # Tworzenie katalogu na wyniki
    for metric in ['Accuracy', 'F1(macro)']:  # Dla metryk Accuracy i F1(macro)
        fig, ax = plt.subplots(figsize=(8, 5))  # Tworzenie wykresu
        ax.bar(df_metrics['Model'], df_metrics[metric])
        ax.set_ylim(0, 1)  # Ustawienie zakresu osi Y
        ax.set_title(f"{metric} porównanie modeli")
        ax.set_ylabel(metric)
        ax.set_xlabel('Model')

        # Dodanie wartości do słupków wykresu
        for idx, val in enumerate(df_metrics[metric]):
            ax.text(idx, val + 0.02, f"{val:.2f}", ha='center')

        plt.tight_layout()  # Optymalizacja układu wykresu
        path = os.path.join(results_dir, f"{metric.lower().replace(' ', '_')}_comparison.png")
        fig.savefig(path, dpi=150)  # Zapisanie wykresu
        plt.close(fig)

        message = f"Zapisano wykres: {path}\n"
        print(message)
        with open(log_file, 'a') as log:
            log.write(message)


# Funkcja do trenowania i oceny modeli
def train_and_evaluate_models(X, y, preprocessor, results_dir='results', log_file='log.txt'):
    os.makedirs(results_dir, exist_ok=True)  # Tworzenie katalogu na wyniki
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21, stratify=y  # Podział na dane treningowe i testowe
    )
    message = "Split z stratify\n"
    print(message)
    with open(log_file, 'a') as log:
        log.write(message)

    # Zestaw modeli do przetestowania
    models = {'Logistic Regression': LogisticRegression(max_iter=1000),
              'Decision Tree': DecisionTreeClassifier(),
              'SVM': SVC()}

    # Określenie kolejności klas
    class_order = [
        'Insufficient_Weight', 'Normal_Weight',
        'Overweight_Level_I', 'Overweight_Level_II',
        'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]

    metrics_list = []  # Lista na metryki modeli
    for name, model in models.items():  # Iterowanie po modelach
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])

        # Walidacja krzyżowa
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        message = f"{name} CV mean: {cv.mean():.4f}\n"
        print(message)
        with open(log_file, 'a') as log:
            log.write(message)

        pipe.fit(X_train, y_train)  # Trenowanie modelu
        y_pred = pipe.predict(X_test)  # Predykcja na danych testowych

        # Raport klasyfikacji
        rep = classification_report(y_test, y_pred)
        message = f"=== {name} ===\n{rep}\n"
        print(message)
        with open(log_file, 'a') as log:
            log.write(message)

        # Obliczenie macierzy pomyłek
        cm = confusion_matrix(y_test, y_pred, labels=class_order)
        plot_confusion_matrix(cm, class_order, results_dir, name, log_file)  # Rysowanie macierzy

        # Zbieranie wyników metryk do analizy
        rep_dict = classification_report(y_test, y_pred, output_dict=True)
        metrics_list.append({
            'Model': name,
            'Accuracy': rep_dict['accuracy'],
            'F1(macro)': rep_dict['macro avg']['f1-score']
        })

    # Tworzenie DataFrame z wynikami metryk
    dfm = pd.DataFrame(metrics_list)
    dfm.to_csv(os.path.join(results_dir, 'model_metrics.csv'), index=False)  # Zapisanie metryk do pliku CSV
    plot_model_comparison(dfm, results_dir, log_file)  # Rysowanie wykresów porównania modeli


# Główna funkcja programu
def main():
    log_file = 'log.txt'  # Plik, do którego będą zapisywane logi
    file_path = 'C:/Users/Kuba/PycharmProjects/MSiD1/ObesityDataSet.csv'  # Ścieżka do pliku z danymi
    target = 'NObeyesdad'  # Kolumna docelowa

    # Rozpoczynamy działanie programu
    df = load_data(file_path, log_file)  # Wczytanie danych
    X, y, pre = preprocess_data(df, target, log_file)  # Przetwarzanie danych
    train_and_evaluate_models(X, y, pre, log_file=log_file)  # Trenowanie i ocena modeli


if __name__ == '__main__':
    main()  # Uruchomienie głównej funkcji
