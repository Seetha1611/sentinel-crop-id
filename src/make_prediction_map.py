"""Carte de prédiction des cultures vs vérité terrain RPG.

Les prédictions affichées sont "hors échantillon" pour la totalité de la
zone : validation croisée spatiale (GroupKFold sur les mêmes blocs
géographiques que train_classifier.py), donc chaque parcelle est prédite par
un modèle qui ne l'a jamais vue à l'entraînement — la carte d'erreurs est
honnête, pas un rappel de ce que le modèle a mémorisé.

Usage : python src/make_prediction_map.py
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

import config
from train_classifier import build_features_full_season, build_spatial_blocks, get_date_columns
from viz_common import (
    CATEGORICAL_DARK, GRID_COLOR, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE,
    canonical_class_order, class_color_map,
)

# Palette de statut (fixe, jamais thématisée) — cf. skill dataviz palette.md.
# Une prédiction est un état (correct/erreur), pas une identité de culture :
# elle porte donc les couleurs de statut, pas la palette catégorielle.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"


def load_data():
    df = pd.read_parquet(config.SIGNATURES_PATH)
    geometry = gpd.read_file(config.PARCELS_GEOMETRY_PATH).set_index("id_parcel")
    gdf = geometry.join(df, how="inner")
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=geometry.crs)


def out_of_fold_predictions(df):
    """Prédiction de chaque parcelle par un modèle qui ne l'a pas vue (ni son bloc)."""
    date_cols = get_date_columns(df)
    X = build_features_full_season(df, date_cols)
    y = df["code_cultu"]
    groups = build_spatial_blocks(df)

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    cv = GroupKFold(n_splits=config.N_CV_FOLDS)
    pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
    return pd.Series(pred, index=df.index, name="predicted")


def _bare_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def plot_maps(gdf, ref, class_order, path):
    color_map = class_color_map(class_order)
    code_to_label = dict(zip(ref["code_cultu"], ref["culture_label"]))

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    gdf.plot(ax=axes[0], color=gdf["code_cultu"].map(color_map), edgecolor=SURFACE, linewidth=0.3)
    axes[0].set_title("Vérité terrain (RPG)", color=INK_PRIMARY, fontsize=13,
                       fontweight="bold", loc="left")

    gdf.plot(ax=axes[1], color=gdf["predicted"].map(color_map), edgecolor=SURFACE, linewidth=0.3)
    axes[1].set_title("Prédiction (RandomForest, hors échantillon)", color=INK_PRIMARY,
                       fontsize=13, fontweight="bold", loc="left")

    correct = gdf["code_cultu"] == gdf["predicted"]
    error_colors = correct.map({True: STATUS_GOOD, False: STATUS_CRITICAL})
    gdf.plot(ax=axes[2], color=error_colors, edgecolor=SURFACE, linewidth=0.3)
    axes[2].set_title(f"Erreurs de classification (précision {correct.mean() * 100:.0f}%)",
                       color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left")

    for ax in axes:
        _bare_axes(ax)

    culture_handles = [
        Patch(facecolor=color_map[code], label=f"{code} — {code_to_label.get(code, code)}")
        for code in class_order
    ]
    fig.legend(
        handles=culture_handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=4, frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
    )

    status_handles = [
        Patch(facecolor=STATUS_GOOD, label="Prédiction correcte"),
        Patch(facecolor=STATUS_CRITICAL, label="Erreur de prédiction"),
    ]
    axes[2].legend(
        handles=status_handles, loc="upper right", frameon=False,
        fontsize=9, labelcolor=INK_SECONDARY,
    )

    fig.suptitle(
        "Carte de prédiction des cultures — Beauce / Loiret, saison 2023-2024",
        color=INK_PRIMARY, fontsize=16, fontweight="bold", x=0.01, ha="left", y=1.04,
    )

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_acreage_comparison(gdf, ref, path):
    """L'assolement au sens strict : surfaces par culture, prédites vs réelles.

    C'est la question business du projet (personne ne demande la culture d'une
    parcelle en particulier ; on demande les surfaces du territoire). Les
    erreurs parcellaires peuvent se compenser ou non à l'agrégation — cette
    figure le mesure au lieu de le supposer.
    """
    code_to_label = dict(zip(ref["code_cultu"], ref["culture_label"]))

    truth = gdf.groupby("code_cultu")["surf_parc"].sum().sort_values(ascending=False)
    predicted = gdf.groupby("predicted")["surf_parc"].sum().reindex(truth.index, fill_value=0)
    delta_pct = (predicted - truth) / truth * 100
    mean_abs_dev = delta_pct.abs().mean()

    n = len(truth)
    y_pos = range(n)
    bar_h = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.barh([y - bar_h / 2 for y in y_pos], truth.values, height=bar_h,
            color=CATEGORICAL_DARK[0], label="Surface réelle (RPG)")
    ax.barh([y + bar_h / 2 for y in y_pos], predicted.values, height=bar_h,
            color=CATEGORICAL_DARK[1], label="Surface prédite (hors échantillon)")

    for i, code in enumerate(truth.index):
        ax.text(truth.values[i] + 30, i - bar_h / 2, f"{truth.values[i]:,.0f} ha".replace(",", " "),
                va="center", fontsize=8.5, color=INK_SECONDARY)
        ax.text(predicted.values[i] + 30, i + bar_h / 2,
                f"{predicted.values[i]:,.0f} ha".replace(",", " "),
                va="center", fontsize=8.5, color=INK_SECONDARY)
        edge = max(truth.values[i], predicted.values[i])
        ax.text(edge + 330, i, f"{delta_pct[code]:+.0f} %",
                va="center", fontsize=9.5, color=INK_MUTED, fontweight="bold")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{code} — {code_to_label.get(code, code)}" for code in truth.index],
        fontsize=9.5, color=INK_SECONDARY,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(truth.max(), predicted.max()) * 1.22)
    ax.set_xlabel("Surface (ha)", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "Assolement prédit vs réel — écart absolu moyen "
        f"{mean_abs_dev:.0f} % par culture",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=14,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame({"reel_ha": truth, "pred_ha": predicted, "ecart_pct": delta_pct})


def main():
    gdf = load_data()
    ref = pd.read_csv(config.RPG_REF_CULTURES_PATH, sep=";").rename(
        columns={"Code": "code_cultu", "Libellé": "culture_label"}
    )

    gdf["predicted"] = out_of_fold_predictions(gdf)
    accuracy = (gdf["code_cultu"] == gdf["predicted"]).mean()
    print(f"Précision hors échantillon (GroupKFold spatial, {len(gdf)} parcelles) : "
          f"{accuracy * 100:.1f}%")

    class_order = canonical_class_order(gdf)
    path = config.OUTPUTS_DIR / "carte_predictions.png"
    plot_maps(gdf, ref, class_order, path)
    print(f"Carte sauvegardée : {path}")

    acreage_path = config.OUTPUTS_DIR / "assolement_pred_vs_reel.png"
    comp = plot_acreage_comparison(gdf, ref, acreage_path)
    print(f"Figure assolement sauvegardée : {acreage_path}")
    print(comp.round(1).to_string())


if __name__ == "__main__":
    main()
