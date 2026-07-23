"""Constitution du cube NDVI Sentinel-2 sur la zone d'étude.

Requête STAC (Microsoft Planetary Computer) -> chargement paresseux du cube
(x, y, time) x (B04, B08, SCL) -> NDVI masqué nuages -> composites médians
par fenêtre de COMPOSITE_FREQ_DAYS jours -> sauvegarde NetCDF.

Usage : python src/build_datacube.py
"""

import odc.stac
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401 (enregistre l'accesseur .rio)
from pyproj import Transformer

import config


def zone_bbox_wgs84():
    """Reprojette la bbox de la zone (EPSG:2154) en lon/lat pour la recherche STAC."""
    transformer = Transformer.from_crs(config.CRS_PROJECTED, config.CRS_WGS84, always_xy=True)
    xmin, ymin, xmax, ymax = config.ZONE_BBOX_2154
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)
    return (lon_min, lat_min, lon_max, lat_max)


def search_items():
    """Recherche les scènes Sentinel-2 L2A sur la zone/période, filtrées par nébulosité."""
    catalog = pystac_client.Client.open(
        config.STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[config.STAC_COLLECTION],
        bbox=zone_bbox_wgs84(),
        datetime=f"{config.SEASON_START}/{config.SEASON_END}",
        query={"eo:cloud_cover": {"lt": config.MAX_CLOUD_COVER_PCT}},
    )
    items = list(search.item_collection())
    if not items:
        raise RuntimeError(
            "Aucune scène Sentinel-2 trouvée pour la zone/période demandée "
            f"(cloud_cover < {config.MAX_CLOUD_COVER_PCT}%)."
        )
    print(f"{len(items)} scènes Sentinel-2 trouvées (cloud_cover < {config.MAX_CLOUD_COVER_PCT}%).")
    return items


def load_datacube(items):
    """Charge B04/B08/SCL en Lambert-93 sur l'emprise de la zone, en paresseux (dask)."""
    xmin, ymin, xmax, ymax = config.ZONE_BBOX_2154
    ds = odc.stac.load(
        items,
        bands=config.BANDS,
        crs=config.CRS_PROJECTED,
        resolution=10,
        x=(xmin, xmax),
        y=(ymin, ymax),
        # SCL est une classification catégorielle : le rééchantillonnage lors
        # de la reprojection UTM31N -> Lambert-93 doit être au plus proche
        # voisin, jamais bilinéaire (qui inventerait des codes de classe).
        resampling={"SCL": "nearest", "*": "bilinear"},
        groupby="solar_day",
        chunks={"time": 1, "x": 1024, "y": 1024},
    )
    return ds


def compute_masked_ndvi(ds):
    """NDVI corrigé du décalage BOA, masqué aux pixels végétation/sol nu du SCL."""
    nodata = (ds["B04"] == 0) | (ds["B08"] == 0)

    red = ds["B04"].astype("float32") + config.S2_BOA_OFFSET
    nir = ds["B08"].astype("float32") + config.S2_BOA_OFFSET
    denom = nir + red
    ndvi = (nir - red) / denom

    # Sur des pixels très sombres (ombres résiduelles, eau non détectée par le
    # SCL...), le décalage BOA peut amener le dénominateur près de 0, voire
    # négatif : le ratio explose alors hors du domaine physique du NDVI
    # ([-1, 1]). On les traite comme non valides plutôt que de clipper une
    # valeur qui n'a de toute façon aucun sens radiométrique.
    valid_scl = ds["SCL"].isin(config.SCL_KEEP_CLASSES)
    valid_denom = denom > 0
    ndvi = ndvi.where(valid_scl & ~nodata & valid_denom)
    ndvi = ndvi.clip(-1, 1)
    ndvi.name = "ndvi"
    return ndvi


def build_composites(ndvi):
    """Médiane NDVI par fenêtre de COMPOSITE_FREQ_DAYS jours sur la saison."""
    freq = f"{config.COMPOSITE_FREQ_DAYS}D"
    composites = ndvi.resample(time=freq, origin=config.SEASON_START).median(
        "time", skipna=True, keep_attrs=True
    )
    return composites


def main():
    items = search_items()
    ds = load_datacube(items)
    print(f"Cube brut (paresseux) : { {k: v for k, v in ds.sizes.items()} }")

    ndvi = compute_masked_ndvi(ds)
    composites = build_composites(ndvi).compute()
    composites = composites.rio.write_crs(config.CRS_PROJECTED)

    valid_frac = composites.notnull().mean(dim=["x", "y"]).values
    print(f"Cube composites : { {k: v for k, v in composites.sizes.items()} }")
    for t, frac in zip(composites["time"].values, valid_frac):
        print(f"  {str(t)[:10]} : {frac * 100:5.1f}% de pixels valides")

    composites.to_netcdf(config.DATACUBE_PATH)
    print(f"Cube sauvegardé : {config.DATACUBE_PATH}")


if __name__ == "__main__":
    main()
