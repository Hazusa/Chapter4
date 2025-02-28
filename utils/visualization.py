import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison(results_file="experiments/results.csv"):
    df = pd.read_csv(results_file, names=["Model", "Accuracy", "F1"])

    plt.figure(figsize=(10, 5))
    for metric in ["Accuracy", "F1"]:
        plt.bar(df["Model"] + "_" + metric, df[metric], alpha=0.6)

    plt.title("Model Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("experiments/comparison.png")