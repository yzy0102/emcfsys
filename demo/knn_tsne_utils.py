from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


DISPLAY_NAME_OVERRIDES = {
    "Autosome": "Autosome",
    "CellWall": "Cell Wall",
    "Cell Wall": "Cell Wall",
    "Chloroplast": "Chloroplast",
    "ER": "ER",
    "Golgi": "Golgi",
    "Goligi": "Golgi",
    "Mito": "Mitochondria",
    "Mitochondria": "Mitochondria",
    "Nucleus": "Nuclear",
    "Nuclear": "Nuclear",
    "Vesicles": "Vesicles",
}

CUSTOM_COLORS = [
    "#dc0239",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#EBD683",
    "#a65628",
    "#f781bf",
    "#B7D28D",
    "#d9b8f1",
]
FIXED_TSNE_TICKS = np.array([-40, -20, 0, 20, 40], dtype=int)


def ensure_tsne_dependencies() -> None:
    missing = []
    for module_name in ("pandas", "seaborn", "matplotlib", "sklearn", "scipy"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "t-SNE visualization requires these packages: "
            f"{', '.join(missing)}. Install them in EMCF_napari, for example: "
            "python -m pip install seaborn scikit-learn scipy pandas matplotlib"
        )


@torch.no_grad()
def extract_dataset_features(
    *,
    feature_extractor,
    dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    feature_extractor.eval()
    all_features = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        features = feature_extractor(images)
        features = F.normalize(features, p=2, dim=1)
        all_features.append(features.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy().astype(int))

    if not all_features:
        raise ValueError("Cannot run t-SNE with an empty dataset.")

    image_paths = [sample[0] for sample in dataset.samples]
    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0),
        image_paths,
    )


def _safe_perplexity(n_samples: int, requested: float | None) -> float:
    if n_samples < 3:
        raise ValueError("t-SNE requires at least 3 samples.")
    if requested is not None:
        return min(float(requested), float(max(1, n_samples - 2)))
    return float(min(30, max(5, (n_samples - 1) // 3), n_samples - 2))


def _display_names(class_names: list[str]) -> list[str]:
    return [DISPLAY_NAME_OVERRIDES.get(name, name) for name in class_names]


def _set_square_limits(ax, x_values: np.ndarray, y_values: np.ndarray) -> None:
    max_abs = float(np.max(np.abs(np.concatenate([x_values, y_values]))))
    half_range = max(max_abs * 1.08, float(np.max(np.abs(FIXED_TSNE_TICKS))) * 1.1)
    ax.set_xlim(-half_range, half_range)
    ax.set_ylim(-half_range, half_range)
    ax.set_aspect("equal", adjustable="box")


def _add_short_axis_ticks(ax) -> None:
    ax.set_xticks(FIXED_TSNE_TICKS)
    ax.set_yticks(FIXED_TSNE_TICKS)
    ax.set_xticklabels([str(value) for value in FIXED_TSNE_TICKS], fontname="Arial")
    ax.set_yticklabels([str(value) for value in FIXED_TSNE_TICKS], fontname="Arial")
    ax.tick_params(
        axis="both",
        which="major",
        bottom=True,
        left=True,
        top=False,
        right=False,
        labelbottom=True,
        labelleft=True,
        direction="out",
        length=3.5,
        width=1.1,
        labelsize=8,
    )


def _add_class_ellipses(ax, df, ordered_names: list[str], palette: dict[str, str]) -> None:
    from matplotlib.patches import Ellipse
    from scipy.spatial.distance import mahalanobis

    for label_name in ordered_names:
        subset = df[df["label_name"] == label_name]
        data = subset[["x", "y"]].to_numpy(dtype=float)
        if len(data) < 5:
            continue

        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        distances = np.array([mahalanobis(point, mean, inv_cov) for point in data])
        threshold = np.percentile(distances, 85)
        filtered_data = data[distances < threshold]
        if len(filtered_data) < 5:
            continue

        mean_f = np.mean(filtered_data, axis=0)
        cov_f = np.cov(filtered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_f)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        ellipse = Ellipse(
            xy=(mean_f[0], mean_f[1]),
            width=float(np.sqrt(eigenvalues[0]) * 3.4),
            height=float(np.sqrt(eigenvalues[1]) * 3.4),
            angle=float(angle),
            edgecolor=palette[label_name],
            facecolor="none",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(ellipse)


def _plot_tsne_with_colorbar(
    *,
    df,
    class_names: list[str],
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["font.family"] = "Arial"
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    import seaborn as sns

    ordered_names = _display_names(class_names)
    colors = CUSTOM_COLORS
    if len(ordered_names) > len(colors):
        colors = sns.color_palette("tab20", n_colors=len(ordered_names)).as_hex()
    palette = {
        label_name: colors[index % len(colors)]
        for index, label_name in enumerate(ordered_names)
    }

    sns.set(style="ticks", font_scale=1.2)
    fig = plt.figure(figsize=(10, 10))
    grid = GridSpec(1, 10, figure=fig)
    ax_main = fig.add_subplot(grid[:, :8])
    ax_legend = fig.add_subplot(grid[:, 8:])

    sns.scatterplot(
        x="x",
        y="y",
        hue="label_name",
        data=df,
        hue_order=ordered_names,
        s=90,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.35,
        ax=ax_main,
        legend=False,
        palette=palette,
    )
    _add_class_ellipses(ax_main, df, ordered_names, palette)
    _set_square_limits(ax_main, df["x"].to_numpy(), df["y"].to_numpy())
    ax_main.set_xlabel("")
    ax_main.set_ylabel("")
    _add_short_axis_ticks(ax_main)

    bar_height = 1.0
    total_height = len(ordered_names) * bar_height
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, total_height)
    ax_legend.axis("off")

    for index, label_name in enumerate(reversed(ordered_names)):
        y_pos = index * bar_height
        color = palette[label_name]
        rect = Rectangle((0.02, y_pos), 0.35, bar_height, color=color)
        ax_legend.add_patch(rect)
        ax_legend.plot(
            [0.42, 0.48],
            [y_pos + bar_height / 2] * 2,
            color="black",
            linewidth=2,
        )
        ax_legend.text(
            0.55,
            y_pos + bar_height / 2,
            label_name,
            va="center",
            fontsize=13,
        )

    frame = Rectangle(
        (0.02, 0.0),
        0.35,
        total_height,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
    )
    ax_legend.add_patch(frame)
    plt.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    svg_path = output_dir / f"{prefix}.svg"
    pdf_path = output_dir / f"{prefix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "pdf": str(pdf_path),
    }


def run_val_tsne(
    *,
    feature_extractor,
    dataset,
    class_names: list[str],
    batch_size: int,
    device: torch.device,
    num_workers: int,
    output_dir: str | Path,
    random_state: int = 42,
    perplexity: float | None = None,
    prefix: str = "val_tsne_with_colorbar",
) -> dict[str, object]:
    ensure_tsne_dependencies()
    import pandas as pd
    from sklearn.manifold import TSNE

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    features, labels, image_paths = extract_dataset_features(
        feature_extractor=feature_extractor,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    np.save(output_path / f"{prefix}_features.npy", features)
    np.save(output_path / f"{prefix}_labels.npy", labels)

    tsne = TSNE(
        n_components=2,
        perplexity=_safe_perplexity(len(labels), perplexity),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    features_tsne = tsne.fit_transform(features)
    display_names = _display_names(class_names)
    label_names = [display_names[int(label)] for label in labels]
    df = pd.DataFrame(
        {
            "x": features_tsne[:, 0],
            "y": features_tsne[:, 1],
            "label": labels.astype(int),
            "label_name": label_names,
            "image_path": image_paths,
        }
    )
    csv_path = output_path / f"{prefix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    figure_paths = _plot_tsne_with_colorbar(
        df=df,
        class_names=class_names,
        output_dir=output_path,
        prefix=prefix,
    )
    return {
        "sample_count": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "perplexity": float(tsne.perplexity),
        "csv": str(csv_path),
        "features_npy": str(output_path / f"{prefix}_features.npy"),
        "labels_npy": str(output_path / f"{prefix}_labels.npy"),
        "figures": figure_paths,
    }
