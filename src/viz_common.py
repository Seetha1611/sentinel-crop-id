"""Constantes visuelles partagées entre tous les scripts de figures du projet.

Centralisé ici pour qu'une même culture porte toujours la même couleur dans
signatures_temporelles.png, matrice_confusion.png et carte_predictions.png —
un lecteur qui a mémorisé "le colza est en rose" doit pouvoir s'y fier partout.
"""

SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID_COLOR = "#2c2c2a"

# Palette catégorielle (variante sombre), ordre fixe validé CVD sur paires
# adjacentes (cf. skill dataviz).
CATEGORICAL_DARK = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
]

# Rampe séquentielle (bleu), ancrage inversé pour surface sombre : les
# faibles valeurs se fondent vers le fond, les fortes valeurs "sortent" en clair.
SEQUENTIAL_DARK = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]


FR_MONTHS = {
    1: "jan", 2: "fév", 3: "mar", 4: "avr", 5: "mai", 6: "jui",
    7: "jul", 8: "aoû", 9: "sep", 10: "oct", 11: "nov", 12: "déc",
}


def french_date_label(ts) -> str:
    return f"{ts.day:02d} {FR_MONTHS[ts.month]}"


def canonical_class_order(df, code_col="code_cultu"):
    """Ordre de référence des classes (par effectif décroissant), utilisé pour
    assigner les couleurs de façon identique dans toutes les figures du projet."""
    return df[code_col].value_counts().index.tolist()


def class_color_map(class_order):
    """code_cultu -> couleur, dans l'ordre canonique (jusqu'à 7 classes)."""
    if len(class_order) > len(CATEGORICAL_DARK):
        raise ValueError(
            f"{len(class_order)} classes > {len(CATEGORICAL_DARK)} couleurs disponibles "
            "dans la palette catégorielle validée."
        )
    return dict(zip(class_order, CATEGORICAL_DARK))
