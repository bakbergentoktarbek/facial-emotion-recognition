# src/evaluate.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import EMOTIONS_EN, EMOTIONS_KZ, MODEL_PATH, REPORTS_DIR
from data import load_datasets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    args = p.parse_args()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.model))[0]

    model = tf.keras.models.load_model(args.model)
    _, _, test_ds = load_datasets()
    names = [EMOTIONS_KZ[e] for e in EMOTIONS_EN]

    # test_ds shuffle=False → суреттер мен белгілердің реті екі өтуде де бірдей
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = model.predict(test_ds, verbose=1)
    y_pred = probs.argmax(axis=1)

    print(f"\nTest accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
    print(classification_report(y_true, y_pred, target_names=names, digits=3))

    # 1) Confusion matrix (жол бойынша нормаланған: әр шын класстың қанша пайызы қайда кетті)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names, vmin=0, vmax=1)
    plt.xlabel("Модель болжамы (predicted)")
    plt.ylabel("Шын эмоция (true)")
    plt.title(f"Confusion matrix — {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"confusion_{tag}.png"), dpi=150)

    # 2) Ең жиі 5 шатастыру
    pairs = [(cm_norm[i, j], names[i], names[j], cm[i, j])
             for i in range(len(names)) for j in range(len(names)) if i != j]
    print("Ең жиі шатастырулар (шын → болжам):")
    for rate, t, pr, n in sorted(pairs, reverse=True)[:5]:
        print(f"  {t:10s} → {pr:10s} {rate:6.1%}  ({n} сурет)")

    # 3) Қате болжамдардың мысалдары — «модель неге қателесті?» деп талқылау үшін
    images = np.concatenate([x.numpy() for x, _ in test_ds])
    wrong = np.where(y_true != y_pred)[0]
    sel = np.random.default_rng(0).choice(wrong, size=min(16, len(wrong)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(8, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, sel):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Шын: {names[y_true[i]]}\nБолжам: {names[y_pred[i]]} ({probs[i].max():.0%})", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"errors_{tag}.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
