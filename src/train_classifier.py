"""Classification des cultures à partir des signatures NDVI.

- Split spatial (jamais aléatoire) par blocs géographiques, pour éviter la
  fuite spatiale entre parcelles voisines quasi identiques.
- RandomForest (baseline) comparé à HistGradientBoosting, en validation
  croisée spatiale (GroupKFold).
- Matrice de confusion + F1 par classe sur les prédictions hors-échantillon
  poolées de la validation croisée spatiale (mêmes 769 parcelles que la
  précision globale de la carte).
- Courbe de performance en fonction de la date : precision atteignable si on
  ne regardait que les données jusqu'à telle date de la saison.

Usage : python src/train_classifier.py
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

import config
from viz_common import (
    CATEGORICAL_DARK, GRID_COLOR, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SEQUENTIAL_DARK,
    SURFACE, canonical_class_order, french_date_label,
)

DERIVED_FEATURE_LABELS = {
    "ndvi_max": "NDVI max",
    "ndvi_min": "NDVI min",
    "ndvi_amplitude": "Amplitude",
    "doy_pic": "Jour du pic",
    "pente_printemps": "Pente printemps",
}


def get_date_columns(df):
    return [c for c in df.columns if c[:2] == "20"]


def build_spatial_blocks(df):
    """Groupe géographique = cellule d'une grille de SPATIAL_BLOCK_SIZE_M mètres."""
    bx = (df["x_centroid"] // config.SPATIAL_BLOCK_SIZE_M).astype(int)
    by = (df["y_centroid"] // config.SPATIAL_BLOCK_SIZE_M).astype(int)
    return (bx.astype(str) + "_" + by.astype(str)).values


def nearest_date_col(date_cols, target_date):
    dates = pd.to_datetime(date_cols)
    idx = np.argmin(np.abs(dates - pd.Timestamp(target_date)))
    return date_cols[idx]


def build_features_full_season(df, date_cols):
    """Série brute + quelques features dérivées simples (max, amplitude, pente printemps)."""
    X = df[date_cols].copy()

    values = df[date_cols].values
    dates = pd.to_datetime(date_cols)

    X["ndvi_max"] = values.max(axis=1)
    X["ndvi_min"] = values.min(axis=1)
    X["ndvi_amplitude"] = X["ndvi_max"] - X["ndvi_min"]

    argmax_idx = values.argmax(axis=1)
    X["doy_pic"] = dates[argmax_idx].dayofyear.values

    early_col = nearest_date_col(date_cols, f"{dates.year.min()}-03-01")
    late_col = nearest_date_col(date_cols, f"{dates.year.max()}-04-15")
    X["pente_printemps"] = df[late_col] - df[early_col]

    return X


def spatial_split(X, y, groups):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def make_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=config.RANDOM_SEED,
        ),
    }


def spatial_cross_validate(models, X, y, groups):
    """GroupKFold : chaque bloc géographique reste entier dans un seul fold."""
    gkf = GroupKFold(n_splits=config.N_CV_FOLDS)
    results = {}
    for name, model in models.items():
        accs, f1s = [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            accs.append(accuracy_score(y.iloc[test_idx], pred))
            f1s.append(f1_score(y.iloc[test_idx], pred, average="macro"))
        results[name] = {
            "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
            "f1_macro_mean": np.mean(f1s), "f1_macro_std": np.std(f1s),
        }
    return results


def performance_by_date(raw_df, date_cols, train_idx, test_idx, y):
    """Précision atteignable en ne connaissant la saison que jusqu'à chaque date.

    Part de la matrice NON interpolée : les trous nuageux sont comblés par
    interpolation *restreinte à la fenêtre déjà observée* à chaque coupure.
    Interpoler sur la saison complète avant de découper (comme le fait le
    parquet principal) ferait fuiter des observations postérieures à la
    coupure — trois composites entièrement nuageux seraient reconstruits
    depuis le futur. Les NaN résiduels (parcelles sans aucune observation en
    tout début de saison) sont imputés à la médiane calculée sur le train.
    """
    rows = []
    for cutoff_i in range(len(date_cols)):
        cols = date_cols[: cutoff_i + 1]
        X = raw_df[cols].interpolate(axis=1, method="linear", limit_direction="both")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

        model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=config.RANDOM_SEED, n_jobs=-1,
        )
        model.fit(X_train, y.iloc[train_idx])
        pred = model.predict(X_test)
        rows.append({
            "date": pd.Timestamp(date_cols[cutoff_i]),
            "accuracy": accuracy_score(y.iloc[test_idx], pred),
            "f1_macro": f1_score(y.iloc[test_idx], pred, average="macro"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Visuels
# --------------------------------------------------------------------------
def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_confusion_matrix(cm, class_order, path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # rappel par ligne (vraie classe)
    n = len(class_order)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq_dark", SEQUENTIAL_DARK)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            value = cm_norm[i, j]
            if value < 0.01:
                continue
            text_color = INK_PRIMARY if value < 0.6 else "#0b0b0b"
            ax.text(
                j, i, f"{value * 100:.0f}%\n({cm[i, j]})",
                ha="center", va="center", fontsize=8.5, color=text_color, linespacing=1.4,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_order, rotation=45, ha="right", color=INK_SECONDARY)
    ax.set_yticklabels(class_order, color=INK_SECONDARY)
    ax.set_xlabel("Culture prédite", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("Culture réelle (RPG)", color=INK_SECONDARY, fontsize=11)
    ax.set_title(
        "Matrice de confusion — rappel par classe (pool-CV, hors échantillon)",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=14,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)
    cbar.set_label("Rappel", color=INK_SECONDARY, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, X_test, y_test, date_cols, path):
    """Importance par permutation sur le test spatial : où le modèle regarde-t-il ?

    Le README annonce un RandomForest "interprétable" — cette figure honore la
    promesse. On permute chaque feature sur le test et on mesure la baisse
    d'accuracy : le profil temporel montre *quand* dans la saison vit
    l'information discriminante.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns)

    date_imp = importances[date_cols]
    derived_imp = importances[[c for c in X_test.columns if c not in date_cols]]

    fig, (ax_dates, ax_derived) = plt.subplots(
        1, 2, figsize=(13, 5.5), dpi=200, width_ratios=[3.2, 1],
    )
    fig.patch.set_facecolor(SURFACE)

    dates = pd.to_datetime(date_cols)
    ax_dates.bar(range(len(date_cols)), date_imp.values, color=CATEGORICAL_DARK[0], width=0.72)
    ax_dates.set_xticks(range(len(date_cols)))
    ax_dates.set_xticklabels([french_date_label(d) for d in dates], rotation=45, ha="right")
    ax_dates.set_ylabel("Baisse d'accuracy après permutation", color=INK_SECONDARY, fontsize=10)
    ax_dates.set_title("Importance des dates d'observation", color=INK_PRIMARY,
                       fontsize=12, fontweight="bold", loc="left")
    ax_dates.set_ylim(0, date_imp.max() * 1.15)

    # Échelle indépendante : ces features valent ~10x moins que la meilleure
    # date, et deux d'entre elles sont même légèrement négatives (permuter la
    # colonne améliore parfois le score par hasard — signe qu'elle n'apporte
    # rien de plus que la série brute déjà présente dans les features).
    derived_sorted = derived_imp.sort_values()
    labels = [DERIVED_FEATURE_LABELS.get(c, c) for c in derived_sorted.index]
    bar_colors = [CATEGORICAL_DARK[0] if v >= 0 else INK_MUTED for v in derived_sorted.values]
    ax_derived.barh(range(len(derived_sorted)), derived_sorted.values,
                    color=bar_colors, height=0.6)
    ax_derived.axvline(0, color=GRID_COLOR, linewidth=1)
    ax_derived.set_yticks(range(len(derived_sorted)))
    ax_derived.set_yticklabels(labels, fontsize=9, color=INK_SECONDARY)
    span = derived_imp.abs().max() * 1.5
    ax_derived.set_xlim(-span, span)
    ax_derived.set_title("Features dérivées (échelle indépendante)", color=INK_PRIMARY,
                         fontsize=12, fontweight="bold", loc="left")

    for ax in (ax_dates, ax_derived):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_dates.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax_derived.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)

    fig.suptitle(
        "Où le modèle regarde-t-il ? Importance par permutation (test spatial)",
        color=INK_PRIMARY, fontsize=13.5, fontweight="bold", x=0.01, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    return importances


def plot_performance_curve(perf_df, path):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    for col, color, label in [
        ("accuracy", CATEGORICAL_DARK[0], "Précision globale (accuracy)"),
        ("f1_macro", CATEGORICAL_DARK[1], "F1 macro (moyenne inter-classes)"),
    ]:
        ax.plot(
            perf_df["date"], perf_df[col], color=color, linewidth=2, label=label,
            solid_capstyle="round", marker="o", markersize=5,
        )

    # Les deux courbes convergent en fin de saison : des étiquettes directes se
    # chevaucheraient. On retombe sur une légende classique (cf. skill dataviz).
    ax.legend(
        loc="lower right", frameon=False, fontsize=10, labelcolor=INK_SECONDARY,
    )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (test spatial)", color=INK_SECONDARY, fontsize=11)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID_COLOR)

    ax.set_title(
        "À partir de quand peut-on prédire l'assolement ? Précision vs date d'observation",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=14,
    )
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_parquet(config.SIGNATURES_PATH)
    date_cols = get_date_columns(df)
    groups = build_spatial_blocks(df)
    y = df["code_cultu"]

    print(f"{len(df)} parcelles, {df['code_cultu'].nunique()} classes, "
          f"{len(np.unique(groups))} blocs géographiques ({config.SPATIAL_BLOCK_SIZE_M} m).")

    # --- Modèle saison complète (features enrichies) ---
    X_full = build_features_full_season(df, date_cols)
    train_idx, test_idx = spatial_split(X_full, y, groups)
    print(f"Split spatial : {len(train_idx)} parcelles train / {len(test_idx)} test "
          f"({len(np.unique(groups[train_idx]))} / {len(np.unique(groups[test_idx]))} blocs).")

    print("\nValidation croisée spatiale (GroupKFold, saison complète) :")
    cv_results = spatial_cross_validate(make_models(), X_full, y, groups)
    for name, res in cv_results.items():
        print(f"  {name:22s} accuracy={res['accuracy_mean']:.3f}±{res['accuracy_std']:.3f}  "
              f"F1_macro={res['f1_macro_mean']:.3f}±{res['f1_macro_std']:.3f}")

    # --- Split unique pour les diagnostics dépendant d'un modèle entraîné
    # (importance par permutation, courbe de performance par date) ---
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_full.iloc[train_idx], y.iloc[train_idx])

    class_order = canonical_class_order(df)

    # --- Matrice de confusion / F1 par classe : prédictions hors-échantillon
    # poolées sur les 5 folds (mêmes 769 parcelles que la précision globale de
    # la carte), plus robustes par classe qu'un split unique à 195 parcelles,
    # et cohérentes avec le F1 macro de la validation croisée du tableau.
    y_pred_cv = cross_val_predict(
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=config.RANDOM_SEED, n_jobs=-1,
        ),
        X_full, y, groups=groups, cv=GroupKFold(n_splits=config.N_CV_FOLDS),
    )
    print(f"\nRapport de classification (RandomForest, pool-CV hors-échantillon, "
          f"{len(df)} parcelles) :")
    print(classification_report(y, y_pred_cv, labels=class_order, zero_division=0))

    cm = confusion_matrix(y, y_pred_cv, labels=class_order)
    plot_confusion_matrix(cm, class_order, config.OUTPUTS_DIR / "matrice_confusion.png")

    importances = plot_feature_importance(
        model, X_full.iloc[test_idx], y.iloc[test_idx], date_cols,
        config.OUTPUTS_DIR / "importance_features.png",
    )
    print("\nTop 8 features (importance par permutation, test spatial) :")
    print(importances.sort_values(ascending=False).head(8).round(4).to_string())

    # --- Courbe de performance en fonction de la date (features brutes seules) ---
    raw_df = pd.read_parquet(config.SIGNATURES_RAW_PATH).loc[df.index]
    perf_df = performance_by_date(raw_df, date_cols, train_idx, test_idx, y)
    print("\nPrécision par date de coupure :")
    print(perf_df.to_string(index=False, formatters={
        "date": lambda d: d.strftime("%Y-%m-%d"),
        "accuracy": "{:.3f}".format,
        "f1_macro": "{:.3f}".format,
    }))
    plot_performance_curve(perf_df, config.OUTPUTS_DIR / "courbe_performance_date.png")

    joblib.dump(
        {"model": model, "feature_columns": X_full.columns.tolist(), "class_order": class_order},
        config.MODEL_OUTPUT_PATH,
    )
    print(f"\nModèle sauvegardé : {config.MODEL_OUTPUT_PATH}")
    print("Figures sauvegardées : matrice_confusion.png, importance_features.png, "
          "courbe_performance_date.png")


if __name__ == "__main__":
    main()
