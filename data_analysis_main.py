import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_csv(file_path, sep=",")


def compute_statistics(df):
    stats = {}

    num_features = df.select_dtypes(include=[np.number])
    stats['numerical'] = pd.DataFrame({
        'mean': num_features.mean(),
        'median': num_features.median(),
        'min': num_features.min(),
        'max': num_features.max(),
        'std': num_features.std(),
        '5th_percentile': num_features.quantile(0.05),
        '95th_percentile': num_features.quantile(0.95),
        'missing_values': num_features.isnull().sum()
    })

    cat_features = df.select_dtypes(exclude=[np.number])
    stats['categorical'] = pd.DataFrame({
        'unique_classes': cat_features.nunique(),
        'missing_values': cat_features.isnull().sum()
    })

    class_proportions = {}
    for col in cat_features:
        class_proportions[col] = cat_features[col].value_counts(normalize=True).to_dict()

    stats['categorical']['class_proportions'] = class_proportions

    return stats


def save_statistics(stats, num_output_file="numerical_statistics.csv", cat_output_file="categorical_statistics.txt"):
    stats['numerical'].to_csv(num_output_file)

    with open(cat_output_file, "w") as f:
        f.write("Categorical Statistics\n")
        f.write("====================\n\n")

        for col, data in stats['categorical'].iterrows():
            f.write(f"Feature: {col}\n")
            f.write(f"Unique Classes: {data['unique_classes']}\n")
            f.write(f"Missing Values: {data['missing_values']}\n")
            f.write("Class Proportions:\n")
            for class_label, proportion in stats['categorical']['class_proportions'].get(col, {}).items():
                f.write(f"  {class_label}: {proportion:.4f}\n")
            f.write("\n")


def create_folders():
    plot_folders = ['boxplots', 'violinplots', 'error_bars', 'histograms', 'conditional_histograms', 'regressions',
                    'heatmaps']
    for folder in plot_folders:
        os.makedirs(folder, exist_ok=True)


def visualize_data(df):
    create_folders()

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    # Kodowanie cech kategorycznych
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Stworzenie heatmapy dla wszystkich cech
    correlation_matrix = df_encoded.corr()  # Obliczanie korelacji po kodowaniu
    plt.figure(figsize=(16, 12))  # Zwiększenie rozmiaru wykresu

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                annot_kws={'size': 8},  # Zmniejszenie rozmiaru czcionki dla anotacji
                cbar_kws={'shrink': 0.8})  # Zmniejszenie paska kolorów
    plt.title('Correlation Heatmap of All Features', fontsize=16)

    # Dostosowanie etykiet osi X i Y
    plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotacja etykiet X
    plt.yticks(rotation=0, ha='right', fontsize=12)  # Rotacja etykiet Y

    plt.tight_layout()  # Dopasowanie układu
    plt.savefig('heatmaps/correlation_heatmap_all_features.png')
    plt.close()

    # Pozostałe wizualizacje (boxplots, violinplots, itp.) pozostają bez zmian
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=df[cat_col], y=df[num_col])
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Boxplot of {num_col} by {cat_col}')
            plt.tight_layout()
            plt.savefig(f'boxplots/boxplot_{num_col}_by_{cat_col}.png')
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.violinplot(x=df[cat_col], y=df[num_col])
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Violinplot of {num_col} by {cat_col}')
            plt.tight_layout()
            plt.savefig(f'violinplots/violinplot_{num_col}_by_{cat_col}.png')
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.pointplot(data=df, x=cat_col, y=num_col, errorbar=("ci", 95), capsize=0.2, linestyle="none")
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Error Bars of {num_col} by {cat_col}')
            plt.tight_layout()
            plt.savefig(f'error_bars/error_bars_{num_col}_by_{cat_col}.png')
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.histplot(df[num_col], kde=True, bins=50)
            plt.title(f'Histogram of {num_col}')
            plt.tight_layout()
            plt.savefig(f'histograms/histogram_{num_col}.png')
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.histplot(df, x=num_col, hue=cat_col, multiple="stack", kde=True, bins=30)
            plt.title(f'Conditional Histogram of {num_col} by {cat_col}')
            plt.tight_layout()
            plt.savefig(f'conditional_histograms/conditional_histogram_{num_col}_by_{cat_col}.png')
            plt.close()

    for num_col_1 in numerical_cols:
        for num_col_2 in numerical_cols:
            if num_col_1 != num_col_2:
                plt.figure(figsize=(12, 6))
                sns.regplot(x=df[num_col_1], y=df[num_col_2], scatter_kws={'s': 20}, line_kws={'color': 'red'})
                plt.title(f'Regression between {num_col_1} and {num_col_2}')
                plt.xlabel(num_col_1)
                plt.ylabel(num_col_2)
                plt.tight_layout()
                plt.savefig(f'regressions/regression_{num_col_1}_vs_{num_col_2}.png')
                plt.close()


def main():
    file_path = "ObesityDataSet.csv"
    df = load_data(file_path)
    save_statistics(compute_statistics(df))
    visualize_data(df)


if __name__ == "__main__":
    main()
