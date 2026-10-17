"""Plot actual epoch records without inventing or smoothing measurements."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import ROOT


def main():
    history = json.loads((ROOT / 'results/training_history_3class.json').read_text())
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'val']:
            ax.plot(epochs, [r[f'{split}_{metric}'] for r in history], label=split)
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.legend()
        ax.grid(alpha=.2)
    for name in history[0]["val_per_class_recall"]:
        axes[2].plot(epochs, [r["val_per_class_recall"][name] for r in history], label=name)
    axes[2].set(xlabel="Epoch", ylabel="Validation recall", ylim=(0, 1.02))
    axes[2].legend(); axes[2].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / 'results/training_curves_3class.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
