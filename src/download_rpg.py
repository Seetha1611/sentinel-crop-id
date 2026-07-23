"""Téléchargement scripté des données RPG (aucune étape manuelle).

- Récupère l'archive régionale RPG millésime courant (config.RPG_DOWNLOAD_URL,
  région Centre-Val de Loire) depuis la Géoplateforme IGN, l'extrait, et
  localise la couche RPG_Parcelles.gpkg qui nous intéresse.
- Récupère la table de correspondance code_cultu -> libellé de culture.

Usage : python src/download_rpg.py
"""

import shutil
import sys
from pathlib import Path

import py7zr
import requests
from tqdm import tqdm

import config


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} déjà présent ({dest.stat().st_size / 1e6:.1f} Mo)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))

    tmp.rename(dest)


def download_rpg_archive() -> Path:
    """Télécharge et extrait l'archive RPG régionale, renvoie le chemin du gpkg parcelles."""
    if config.RPG_GPKG_PATH.exists():
        print(f"[skip] {config.RPG_GPKG_PATH} déjà présent")
        return config.RPG_GPKG_PATH

    archive_path = config.DATA_RAW_DIR / "RPG_region.7z"
    _download(config.RPG_DOWNLOAD_URL, archive_path)

    extract_dir = config.DATA_RAW_DIR / "_rpg_extract_tmp"
    print("Extraction de l'archive 7z...")
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=extract_dir)

    matches = list(extract_dir.rglob("RPG_Parcelles.gpkg"))
    if not matches:
        raise FileNotFoundError(
            f"RPG_Parcelles.gpkg introuvable dans l'archive extraite ({extract_dir})"
        )

    shutil.move(str(matches[0]), config.RPG_GPKG_PATH)
    shutil.rmtree(extract_dir)
    archive_path.unlink()

    return config.RPG_GPKG_PATH


def download_ref_cultures() -> Path:
    """Télécharge la table de correspondance code_cultu -> libellé de culture."""
    _download(config.RPG_REF_CULTURES_URL, config.RPG_REF_CULTURES_PATH)
    return config.RPG_REF_CULTURES_PATH


if __name__ == "__main__":
    gpkg = download_rpg_archive()
    ref = download_ref_cultures()
    print(f"RPG parcelles : {gpkg}")
    print(f"Référentiel cultures : {ref}")
    sys.exit(0)
