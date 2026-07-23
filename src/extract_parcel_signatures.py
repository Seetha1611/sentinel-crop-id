"""Extraction des signatures NDVI par parcelle RPG.

Pour chaque parcelle agricole de la zone : moyenne spatiale du NDVI à chaque
pas de temps du cube (rasterisation des parcelles + groupby, plus rapide que
des zonal stats parcelle par parcelle sur ~1000 parcelles x 21 dates).

Livrable : DataFrame parcelles x pas_de_temps, avec code_cultu / libellé /
surface en métadonnées, sauvegardé en Parquet.

Usage : python src/extract_parcel_signatures.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 (enregistre l'accesseur .rio)
import xarray as xr
from rasterio.features import rasterize
from shapely.geometry import box

import config


def load_datacube():
    ds = xr.open_dataset(config.DATACUBE_PATH)["ndvi"]
    # Le round-trip NetCDF ne préserve pas toujours le CRS de façon exploitable
    # par rioxarray -> on le réaffirme explicitement (même valeur qu'à l'écriture).
    ds = ds.rio.write_crs(config.CRS_PROJECTED)
    return ds


def load_parcels(ndvi):
    # Léger padding sur la requête bbox : la grille NDVI (alignée par odc-stac)
    # peut légèrement déborder de ZONE_BBOX_2154. On clippe ensuite aux bornes
    # exactes du cube.
    xmin, ymin, xmax, ymax = config.ZONE_BBOX_2154
    pad = 100
    gdf = gpd.read_file(
        config.RPG_GPKG_PATH, bbox=(xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    )
    gdf = gdf.clip(box(*ndvi.rio.bounds()))
    return gdf.reset_index(drop=True)


def load_culture_labels():
    ref = pd.read_csv(config.RPG_REF_CULTURES_PATH, sep=";")
    ref = ref.rename(columns={"Code": "code_cultu", "Libellé": "culture_label"})
    return ref[["code_cultu", "culture_label"]]


def filter_parcels(gdf):
    n0 = len(gdf)
    gdf = gdf[~gdf["code_cultu"].isin(config.NON_CROP_CODE_CULTU)]
    gdf = gdf[gdf["surf_parc"] >= config.MIN_PARCEL_AREA_HA]
    print(f"Filtre codes non-cultures + surface >= {config.MIN_PARCEL_AREA_HA} ha : "
          f"{n0} -> {len(gdf)} parcelles")
    return gdf.reset_index(drop=True)


def apply_negative_buffer(gdf):
    n0 = len(gdf)
    buffered = gdf.copy()
    buffered["geometry"] = buffered.geometry.buffer(config.PARCEL_NEGATIVE_BUFFER_M)
    buffered = buffered[~buffered.geometry.is_empty]
    print(f"Buffer négatif {config.PARCEL_NEGATIVE_BUFFER_M} m : "
          f"{n0} -> {len(buffered)} parcelles (les plus petites/étroites disparaissent)")
    return buffered.reset_index(drop=True)


def select_target_classes(gdf, ref):
    counts = gdf["code_cultu"].value_counts()
    eligible = counts[counts >= config.MIN_PARCELS_PER_CLASS]
    top_classes = eligible.head(config.N_TARGET_CLASSES).index.tolist()

    print(f"\nClasses retenues ({len(top_classes)} parmi {len(counts)} codes présents, "
          f"seuil >= {config.MIN_PARCELS_PER_CLASS} parcelles) :")
    for code in top_classes:
        label_rows = ref.loc[ref["code_cultu"] == code, "culture_label"]
        label = label_rows.iloc[0] if len(label_rows) else code
        print(f"  {code:4s} {label:45s} n={counts[code]}")

    return gdf[gdf["code_cultu"].isin(top_classes)].reset_index(drop=True)


def rasterize_parcels(gdf, ndvi):
    """Rasterise les parcelles sur la grille du cube NDVI (id 1-based, 0 = hors parcelle)."""
    transform = ndvi.rio.transform()
    out_shape = (ndvi.sizes["y"], ndvi.sizes["x"])
    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    parcel_raster = rasterize(
        shapes, out_shape=out_shape, transform=transform, fill=0, dtype="int32"
    )
    n_with_pixels = len(np.unique(parcel_raster)) - 1  # -1 pour le fond (0)
    if n_with_pixels < len(gdf):
        print(f"Attention : {len(gdf) - n_with_pixels} parcelle(s) trop petite(s) "
              f"pour couvrir un pixel de 10 m après buffer -> aucune valeur NDVI.")
    return parcel_raster


def extract_mean_timeseries(gdf, ndvi, parcel_raster):
    """Moyenne NDVI par parcelle et par pas de temps (rasterisation + groupby)."""
    mask = parcel_raster > 0
    flat_ids = parcel_raster[mask]

    dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in ndvi["time"].values]
    values = ndvi.values  # (time, y, x)

    columns = {}
    for t_idx, date in enumerate(dates):
        flat_vals = values[t_idx][mask]
        means = pd.Series(flat_vals).groupby(flat_ids).mean()
        columns[date] = means

    wide = pd.DataFrame(columns)
    wide = wide.reindex(range(1, len(gdf) + 1))  # 1 ligne par parcelle, même sans pixel
    wide.index = gdf["id_parcel"].values
    wide.index.name = "id_parcel"
    return wide


def interpolate_gaps(wide):
    """Interpolation linéaire par parcelle des dates entièrement/partiellement nuageuses."""
    n_empty_before = wide.isna().all(axis=1).sum()
    filled = wide.interpolate(axis=1, method="linear", limit_direction="both")
    n_empty_after = filled.isna().all(axis=1).sum()
    print(f"\nInterpolation temporelle : {n_empty_before} parcelle(s) sans aucune donnée avant, "
          f"{n_empty_after} après (à exclure si > 0).")
    return filled[~filled.isna().all(axis=1)]


def main():
    ndvi = load_datacube()
    ref = load_culture_labels()

    gdf = load_parcels(ndvi)
    print(f"Parcelles RPG dans la zone : {len(gdf)}")

    # Géométrie d'origine (avant buffer négatif) conservée à part, pour un
    # affichage fidèle des contours de parcelles dans make_prediction_map.py.
    original_geometry = gdf[["id_parcel", "geometry"]].copy()

    gdf = filter_parcels(gdf)
    gdf = apply_negative_buffer(gdf)
    gdf = select_target_classes(gdf, ref)

    parcel_raster = rasterize_parcels(gdf, ndvi)
    wide_raw = extract_mean_timeseries(gdf, ndvi, parcel_raster)
    wide = interpolate_gaps(wide_raw)

    centroids = gdf.geometry.centroid
    gdf["x_centroid"] = centroids.x
    gdf["y_centroid"] = centroids.y

    meta = gdf[["id_parcel", "code_cultu", "surf_parc", "x_centroid", "y_centroid"]].merge(
        ref, on="code_cultu", how="left"
    )
    meta = meta.set_index("id_parcel")
    signatures = meta.join(wide, how="inner")

    signatures.to_parquet(config.SIGNATURES_PATH)
    print(f"\nSignatures sauvegardées : {config.SIGNATURES_PATH} "
          f"({len(signatures)} parcelles x {wide.shape[1]} pas de temps)")
    print(signatures["code_cultu"].value_counts())

    # Matrice brute (NaN conservés), mêmes parcelles : la courbe "précision vs
    # date" de train_classifier.py doit interpoler à l'intérieur de chaque
    # fenêtre de coupure, jamais avec des observations postérieures.
    wide_raw.loc[signatures.index].to_parquet(config.SIGNATURES_RAW_PATH)
    print(f"Signatures brutes (non interpolées) : {config.SIGNATURES_RAW_PATH}")

    kept_geometry = original_geometry[original_geometry["id_parcel"].isin(signatures.index)]
    kept_geometry.to_file(config.PARCELS_GEOMETRY_PATH, driver="GPKG")
    print(f"Géométries sauvegardées : {config.PARCELS_GEOMETRY_PATH}")


if __name__ == "__main__":
    main()
