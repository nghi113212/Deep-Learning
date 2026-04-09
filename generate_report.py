# -*- coding: utf-8 -*-
"""
Generate training report with accuracy/loss charts and best results summary.
Reads checkpoint logs and produces a PDF report.
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

CHECKPOINT_DIR = "./checkpoints"


def parse_plain_metric(filepath):
    """Parse files with format: epoch_num value"""
    epochs, values = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            epoch = int(parts[0])
            values_str = parts[1]
            match = re.search(r"tensor\(([\d.]+)", values_str)
            if match:
                val = float(match.group(1))
            else:
                val = float(values_str)
            epochs.append(epoch)
            values.append(val)
    return np.array(epochs), np.array(values)


def parse_log_file(filepath):
    """Parse summary log files with format:
    date time Epoch X/Y summary: loss_train=..., acc_train=...%, loss_val=..., acc_val=...% (best: ...% @ epoch ...)
    """
    epochs, loss_train, acc_train, loss_val, acc_val = [], [], [], [], []
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+summary:\s+"
        r"loss_train=([\d.]+),\s+"
        r"acc_train=([\d.]+)%,\s+"
        r"loss_val=([\d.]+),\s+"
        r"acc_val=([\d.]+)%"
    )
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                loss_train.append(float(m.group(2)))
                acc_train.append(float(m.group(3)))
                loss_val.append(float(m.group(4)))
                acc_val.append(float(m.group(5)))
    return {
        "epochs": np.array(epochs),
        "loss_train": np.array(loss_train),
        "acc_train": np.array(acc_train),
        "loss_val": np.array(loss_val),
        "acc_val": np.array(acc_val),
    }


def load_dataset_d(size_label):
    """Load Dataset D (5-class) data from separate metric files + log file."""
    if size_label == "32x32":
        base = os.path.join(CHECKPOINT_DIR, "M1_datasets32_5_")
        log_file = os.path.join(base, "M1_datasets32_5.txt")
    else:
        base = os.path.join(CHECKPOINT_DIR, "M1_datasets224_5_")
        log_file = os.path.join(base, "M1_datasets224_5.txt")

    data = parse_log_file(log_file)

    loss_epochs, loss_values = parse_plain_metric(os.path.join(base, "Loss_plot.txt"))
    data["loss_train_from_file"] = loss_values

    _, train_acc = parse_plain_metric(os.path.join(base, "train_prec1.txt"))
    _, val_acc = parse_plain_metric(os.path.join(base, "val_prec1.txt"))
    data["train_prec1"] = train_acc
    data["val_prec1"] = val_acc

    return data


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "lines.linewidth": 1.8,
    })


def plot_accuracy_curves(ax, epochs, train_acc, val_acc, title):
    ax.plot(epochs, train_acc, label="Train Accuracy", color="#2196F3", alpha=0.85)
    ax.plot(epochs, val_acc, label="Val Accuracy", color="#F44336", alpha=0.85)

    best_idx = np.argmax(val_acc)
    best_epoch = epochs[best_idx]
    best_val = val_acc[best_idx]
    ax.scatter([best_epoch], [best_val], color="#F44336", s=80, zorder=5,
               edgecolors="black", linewidths=0.8)
    ax.annotate(f"Best: {best_val:.2f}%\n(epoch {best_epoch})",
                xy=(best_epoch, best_val),
                xytext=(15, -15), textcoords="offset points",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right")


def plot_loss_curves(ax, epochs, loss_train, title, loss_val=None):
    ax.plot(epochs, loss_train, label="Train Loss", color="#4CAF50", alpha=0.85)
    if loss_val is not None:
        ax.plot(epochs, loss_val, label="Val Loss", color="#FF9800", alpha=0.85)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")


def main():
    set_style()

    # --- Load all data ---
    d32 = load_dataset_d("32x32")
    d224 = load_dataset_d("224x224")
    cifar10 = parse_log_file(os.path.join(CHECKPOINT_DIR, "CIFAR10_M1_Net", "FIE02.txt"))
    cifar100 = parse_log_file(os.path.join(CHECKPOINT_DIR, "CIFAR100_M1_Net", "CIFAR100_dataset.txt"))

    output_pdf = "Training_Report.pdf"
    with PdfPages(output_pdf) as pdf:

        # =====================================================================
        # PAGE 1: Title + Summary Table
        # =====================================================================
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Deep Learning Training Report\nModel M1 - All Experiments",
                      fontsize=20, fontweight="bold", y=0.92)

        ax = fig.add_axes([0.1, 0.15, 0.8, 0.65])
        ax.axis("off")

        experiments = [
            ("Dataset D (32x32)", "5", 100,
             f"{d32['acc_train'][-1]:.2f}%",
             f"{d32['acc_val'].max():.2f}%",
             f"{d32['epochs'][np.argmax(d32['acc_val'])]}",
             f"{d32['loss_train'][-1]:.5f}"),
            ("Dataset D (224x224)", "5", 100,
             f"{d224['acc_train'][-1]:.2f}%",
             f"{d224['acc_val'].max():.2f}%",
             f"{d224['epochs'][np.argmax(d224['acc_val'])]}",
             f"{d224['loss_train'][-1]:.5f}"),
            ("CIFAR-10", "10", 200,
             f"{cifar10['acc_train'][-1]:.2f}%",
             f"{cifar10['acc_val'].max():.2f}%",
             f"{cifar10['epochs'][np.argmax(cifar10['acc_val'])]}",
             f"{cifar10['loss_train'][-1]:.5f}"),
            ("CIFAR-100", "100", 200,
             f"{cifar100['acc_train'][-1]:.2f}%",
             f"{cifar100['acc_val'].max():.2f}%",
             f"{cifar100['epochs'][np.argmax(cifar100['acc_val'])]}",
             f"{cifar100['loss_train'][-1]:.5f}"),
        ]

        col_labels = ["Experiment", "Classes", "Epochs",
                       "Final Train Acc", "Best Val Acc", "Best Epoch", "Final Train Loss"]
        table_data = [list(row) for row in experiments]

        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.0)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1565C0")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#E3F2FD")
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("#BBDEFB")

        ax.set_title("Summary of All Experiments", fontsize=14,
                      fontweight="bold", pad=20)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PAGE 2: Dataset D - 32x32 (Accuracy + Loss)
        # =====================================================================
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("Dataset D - 32x32 Resolution (5 Classes, 100 Epochs)",
                      fontsize=16, fontweight="bold")

        plot_accuracy_curves(axes[0], d32["epochs"], d32["acc_train"], d32["acc_val"],
                             "Training vs Validation Accuracy (32x32)")
        plot_loss_curves(axes[1], d32["epochs"], d32["loss_train"],
                         "Training vs Validation Loss (32x32)", d32["loss_val"])

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PAGE 3: Dataset D - 224x224 (Accuracy + Loss)
        # =====================================================================
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("Dataset D - 224x224 Resolution (5 Classes, 100 Epochs)",
                      fontsize=16, fontweight="bold")

        plot_accuracy_curves(axes[0], d224["epochs"], d224["acc_train"], d224["acc_val"],
                             "Training vs Validation Accuracy (224x224)")
        plot_loss_curves(axes[1], d224["epochs"], d224["loss_train"],
                         "Training vs Validation Loss (224x224)", d224["loss_val"])

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PAGE 4: 32x32 vs 224x224 Comparison
        # =====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Dataset D Comparison: 32x32 vs 224x224",
                      fontsize=16, fontweight="bold")

        # Train accuracy comparison
        axes[0, 0].plot(d32["epochs"], d32["acc_train"],
                        label="32x32", color="#2196F3", alpha=0.85)
        axes[0, 0].plot(d224["epochs"], d224["acc_train"],
                        label="224x224", color="#E91E63", alpha=0.85)
        axes[0, 0].set_title("Training Accuracy Comparison", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].legend()

        # Val accuracy comparison
        axes[0, 1].plot(d32["epochs"], d32["acc_val"],
                        label="32x32", color="#2196F3", alpha=0.85)
        axes[0, 1].plot(d224["epochs"], d224["acc_val"],
                        label="224x224", color="#E91E63", alpha=0.85)
        axes[0, 1].set_title("Validation Accuracy Comparison", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()

        # Train loss comparison
        axes[1, 0].plot(d32["epochs"], d32["loss_train"],
                        label="32x32", color="#2196F3", alpha=0.85)
        axes[1, 0].plot(d224["epochs"], d224["loss_train"],
                        label="224x224", color="#E91E63", alpha=0.85)
        axes[1, 0].set_title("Training Loss Comparison", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()

        # Val loss comparison
        axes[1, 1].plot(d32["epochs"], d32["loss_val"],
                        label="32x32", color="#2196F3", alpha=0.85)
        axes[1, 1].plot(d224["epochs"], d224["loss_val"],
                        label="224x224", color="#E91E63", alpha=0.85)
        axes[1, 1].set_title("Validation Loss Comparison", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PAGE 5: Best Results on Dataset D (32x32 and 224x224)
        # =====================================================================
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Best Testing Results on Dataset D",
                      fontsize=18, fontweight="bold", y=0.95)

        gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                               left=0.08, right=0.95, top=0.88, bottom=0.08)

        # Bar chart comparing best val accuracies
        ax_bar = fig.add_subplot(gs[0, :])
        resolutions = ["32x32", "224x224"]
        best_accs = [d32["acc_val"].max(), d224["acc_val"].max()]
        best_epochs = [d32["epochs"][np.argmax(d32["acc_val"])],
                       d224["epochs"][np.argmax(d224["acc_val"])]]
        colors = ["#2196F3", "#E91E63"]
        bars = ax_bar.bar(resolutions, best_accs, color=colors, width=0.5,
                          edgecolor="black", linewidth=0.8)
        for bar, acc, ep in zip(bars, best_accs, best_epochs):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{acc:.2f}%\n(epoch {ep})",
                        ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax_bar.set_ylim(0, 100)
        ax_bar.set_ylabel("Best Validation Accuracy (%)")
        ax_bar.set_title("Best Validation Accuracy: 32x32 vs 224x224", fontweight="bold")

        # Detailed metrics table for 32x32
        ax_t1 = fig.add_subplot(gs[1, 0])
        ax_t1.axis("off")
        best_idx_32 = np.argmax(d32["acc_val"])
        table_data_32 = [
            ["Resolution", "32x32"],
            ["Best Val Accuracy", f"{d32['acc_val'].max():.2f}%"],
            ["Best Epoch", f"{d32['epochs'][best_idx_32]}"],
            ["Train Acc @ Best", f"{d32['acc_train'][best_idx_32]:.2f}%"],
            ["Train Loss @ Best", f"{d32['loss_train'][best_idx_32]:.5f}"],
            ["Val Loss @ Best", f"{d32['loss_val'][best_idx_32]:.5f}"],
            ["Final Val Accuracy", f"{d32['acc_val'][-1]:.2f}%"],
        ]
        t1 = ax_t1.table(cellText=table_data_32, loc="center", cellLoc="center")
        t1.auto_set_font_size(False)
        t1.set_fontsize(10)
        t1.scale(1.0, 1.8)
        for (row, col), cell in t1.get_celld().items():
            if col == 0:
                cell.set_facecolor("#E3F2FD")
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#BBDEFB")
        ax_t1.set_title("32x32 Best Results", fontweight="bold", fontsize=12)

        # Detailed metrics table for 224x224
        ax_t2 = fig.add_subplot(gs[1, 1])
        ax_t2.axis("off")
        best_idx_224 = np.argmax(d224["acc_val"])
        table_data_224 = [
            ["Resolution", "224x224"],
            ["Best Val Accuracy", f"{d224['acc_val'].max():.2f}%"],
            ["Best Epoch", f"{d224['epochs'][best_idx_224]}"],
            ["Train Acc @ Best", f"{d224['acc_train'][best_idx_224]:.2f}%"],
            ["Train Loss @ Best", f"{d224['loss_train'][best_idx_224]:.5f}"],
            ["Val Loss @ Best", f"{d224['loss_val'][best_idx_224]:.5f}"],
            ["Final Val Accuracy", f"{d224['acc_val'][-1]:.2f}%"],
        ]
        t2 = ax_t2.table(cellText=table_data_224, loc="center", cellLoc="center")
        t2.auto_set_font_size(False)
        t2.set_fontsize(10)
        t2.scale(1.0, 1.8)
        for (row, col), cell in t2.get_celld().items():
            if col == 0:
                cell.set_facecolor("#FCE4EC")
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#F8BBD0")
        ax_t2.set_title("224x224 Best Results", fontweight="bold", fontsize=12)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PAGE 6: CIFAR-10 (Accuracy + Loss)
        # =====================================================================
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("CIFAR-10 Dataset (10 Classes, 200 Epochs)",
                      fontsize=16, fontweight="bold")

        plot_accuracy_curves(axes[0], cifar10["epochs"], cifar10["acc_train"],
                             cifar10["acc_val"],
                             "Training vs Validation Accuracy (CIFAR-10)")
        plot_loss_curves(axes[1], cifar10["epochs"], cifar10["loss_train"],
                         "Training vs Validation Loss (CIFAR-10)", cifar10["loss_val"])

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PAGE 7: CIFAR-100 (Accuracy + Loss)
        # =====================================================================
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("CIFAR-100 Dataset (100 Classes, 200 Epochs)",
                      fontsize=16, fontweight="bold")

        plot_accuracy_curves(axes[0], cifar100["epochs"], cifar100["acc_train"],
                             cifar100["acc_val"],
                             "Training vs Validation Accuracy (CIFAR-100)")
        plot_loss_curves(axes[1], cifar100["epochs"], cifar100["loss_train"],
                         "Training vs Validation Loss (CIFAR-100)", cifar100["loss_val"])

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PAGE 8: All Experiments - Validation Accuracy Comparison
        # =====================================================================
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("All Experiments - Validation Accuracy Overview",
                      fontsize=16, fontweight="bold")

        # Top: All val accuracy curves
        axes[0].plot(d32["epochs"], d32["acc_val"],
                     label=f"Dataset D 32x32 (best {d32['acc_val'].max():.2f}%)",
                     color="#2196F3", alpha=0.85)
        axes[0].plot(d224["epochs"], d224["acc_val"],
                     label=f"Dataset D 224x224 (best {d224['acc_val'].max():.2f}%)",
                     color="#E91E63", alpha=0.85)
        axes[0].plot(cifar10["epochs"], cifar10["acc_val"],
                     label=f"CIFAR-10 (best {cifar10['acc_val'].max():.2f}%)",
                     color="#4CAF50", alpha=0.85)
        axes[0].plot(cifar100["epochs"], cifar100["acc_val"],
                     label=f"CIFAR-100 (best {cifar100['acc_val'].max():.2f}%)",
                     color="#FF9800", alpha=0.85)
        axes[0].set_title("Validation Accuracy - All Experiments", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].legend(loc="lower right")

        # Bottom: Bar chart of best accuracies
        exp_names = ["D (32x32)", "D (224x224)", "CIFAR-10", "CIFAR-100"]
        best_vals = [d32["acc_val"].max(), d224["acc_val"].max(),
                     cifar10["acc_val"].max(), cifar100["acc_val"].max()]
        bar_colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
        bars = axes[1].bar(exp_names, best_vals, color=bar_colors, width=0.55,
                           edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, best_vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.2f}%", ha="center", va="bottom",
                         fontweight="bold", fontsize=11)
        axes[1].set_ylim(0, 105)
        axes[1].set_ylabel("Best Validation Accuracy (%)")
        axes[1].set_title("Best Validation Accuracy Comparison", fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nReport generated successfully: {os.path.abspath(output_pdf)}")
    print(f"Total pages: 8")
    print(f"\nBest results on Dataset D:")
    print(f"  32x32  -> {d32['acc_val'].max():.2f}% (epoch {d32['epochs'][np.argmax(d32['acc_val'])]})")
    print(f"  224x224 -> {d224['acc_val'].max():.2f}% (epoch {d224['epochs'][np.argmax(d224['acc_val'])]})")


if __name__ == "__main__":
    main()
