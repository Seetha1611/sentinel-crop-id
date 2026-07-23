"""LE visuel signature du projet : NDVI moyen ± écart-type par culture.

Doit permettre de voir à l'œil nu que le colza, le blé et le maïs (et les
autres cultures cibles) suivent des trajectoires distinctes au cours de la
saison. Thème sombre cohérent avec le reste du portfolio.

Usage : python src/plot_signatures.py
"""

import matplotlib.pyplot as plt
import pandas as pd

import config
from viz_common import (
    CATEGORICAL_DARK, GRID_COLOR, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE,
    canonical_class_order, french_date_label,
)


def load_class_stats():
    df = pd.read_parquet(config.SIGNATURES_PATH)
    date_cols = [c for c in df.columns if c[:2] == "20"]

    stats_by_code = {}
    for code, group in df.groupby("code_cultu"):
        label = group["culture_label"].iloc[0]
        mean = group[date_cols].mean()
        std = group[date_cols].std()
        stats_by_code[code] = {
            "code_cultu": code,
            "culture_label": label,
            "n_parcelles": len(group),
            "surf_ha": group["surf_parc"].sum(),
            "mean": mean,
            "std": std,
        }

    # Ordre canonique (effectif décroissant) : même ordre, donc même couleur
    # par culture, que la matrice de confusion et la carte de prédiction.
    order = canonical_class_order(df)
    stats = [stats_by_code[code] for code in order]
    dates = pd.to_datetime(date_cols)
    return stats, dates


def plot_signatures(stats, dates, path):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for stat, color in zip(stats, CATEGORICAL_DARK):
        mean = stat["mean"].values
        std = stat["std"].values
        ax.plot(
            dates, mean,
            color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            label=f"{stat['code_cultu']} — {stat['culture_label']} (n={stat['n_parcelles']})",
        )
        ax.fill_between(dates, mean - std, mean + std, color=color, alpha=0.07, linewidth=0)

    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("NDVI moyen ± écart-type", color=INK_SECONDARY, fontsize=11)

    ax.set_xticks(dates)
    ax.set_xticklabels([french_date_label(d) for d in dates], rotation=45, ha="right")

    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)

    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID_COLOR)

    ax.set_title(
        "Signatures temporelles NDVI par culture — Beauce / Loiret, saison 2023-2024",
        color=INK_PRIMARY, fontsize=14, fontweight="bold", loc="left", pad=16,
    )

    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
        frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
    )
    for line in legend.get_lines():
        line.set_linewidth(3)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    stats, dates = load_class_stats()
    out_path = config.OUTPUTS_DIR / "signatures_temporelles.png"
    plot_signatures(stats, dates, out_path)
    print(f"Figure sauvegardée : {out_path}")


if __name__ == "__main__":
    main()
