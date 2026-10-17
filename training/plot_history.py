"""Plot actual epoch records without inventing or smoothing measurements."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import ROOT


def main():
    history = json.loads((ROOT / 'results/training_history.json').read_text())
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'val']:
            ax.plot(epochs, [r[f'{split}_{metric}'] for r in history], label=split)
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / 'results/training_curves.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
