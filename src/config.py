"""Paramètres centraux du pipeline : zone d'étude, dates, CRS, chemins.

Toute constante réutilisée par plusieurs scripts vit ici pour que l'ensemble
du pipeline (cube de données -> signatures -> classification -> carte) reste
cohérent si on change de zone, de millésime ou d'échelle.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
OUTPUTS_DIR = ROOT_DIR / "outputs"

for _dir in (DATA_RAW_DIR, DATA_INTERIM_DIR, OUTPUTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DATACUBE_PATH = DATA_INTERIM_DIR / "ndvi_datacube.nc"
SIGNATURES_PATH = DATA_INTERIM_DIR / "parcel_signatures.parquet"
# Même matrice avant interpolation temporelle (NaN conservés) : indispensable
# pour la courbe "précision vs date", où interpoler sur toute la saison ferait
# fuiter de l'information postérieure à la date de coupure.
SIGNATURES_RAW_PATH = DATA_INTERIM_DIR / "parcel_signatures_raw.parquet"
PARCELS_GEOMETRY_PATH = DATA_INTERIM_DIR / "parcels_geometry.gpkg"
RPG_GPKG_PATH = DATA_RAW_DIR / "RPG_Parcelles.gpkg"
RPG_REF_CULTURES_PATH = DATA_RAW_DIR / "ref_cultures.csv"

# --------------------------------------------------------------------------
# CRS commun à tout le pipeline
# --------------------------------------------------------------------------
# Lambert-93 : CRS légal pour la France métropolitaine, celui dans lequel le
# RPG est livré nativement. On reprojette Sentinel-2 (UTM 31N natif) dedans
# une seule fois, plutôt que de jongler entre deux CRS.
CRS_PROJECTED = "EPSG:2154"
CRS_WGS84 = "EPSG:4326"

# --------------------------------------------------------------------------
# Zone d'étude : Beauce / Loiret, secteur d'Artenay
# --------------------------------------------------------------------------
# Centre indicatif (grandes cultures très lisibles : blé, orge, colza, maïs,
# betterave). Coordonnées en Lambert-93, calculées depuis 48.10°N, 1.90°E.
ZONE_CENTER_X = 618_120  # mètres, EPSG:2154
ZONE_CENTER_Y = 6_778_307  # mètres, EPSG:2154

# Demi-largeur de la zone carrée. Commencer à 5 km (zone 10x10 km) pour
# valider le pipeline de bout en bout ; monter à 10 km (20x20 km) ensuite.
ZONE_HALF_WIDTH_M = 5_000

ZONE_BBOX_2154 = (
    ZONE_CENTER_X - ZONE_HALF_WIDTH_M,
    ZONE_CENTER_Y - ZONE_HALF_WIDTH_M,
    ZONE_CENTER_X + ZONE_HALF_WIDTH_M,
    ZONE_CENTER_Y + ZONE_HALF_WIDTH_M,
)

# --------------------------------------------------------------------------
# Millésime RPG et fenêtre temporelle Sentinel-2
# --------------------------------------------------------------------------
# RPG 2024 = dernier millésime publié (révision IGN de novembre 2025).
# Le RPG 2024 décrit les parcelles déclarées en 2024, arrêtées au 1er
# janvier 2025 -> on l'associe à la saison culturale nov. 2023 -> août 2024
# (semis d'hiver jusqu'à la moisson).
RPG_MILLESIME = 2024
RPG_REGION_CODE = "R24"  # Centre-Val de Loire (couvre le Loiret)
RPG_DOWNLOAD_URL = (
    "https://data.geopf.fr/telechargement/download/RPG/"
    "RPG_3-0__GPKG_LAMB93_R24_2024-01-01/"
    "RPG_3-0__GPKG_LAMB93_R24_2024-01-01.7z"
)
# Table de correspondance code_cultu -> libellé de culture. Les liens
# statiques historiques (geoservices.ign.fr/sites/default/files/...) ont été
# désactivés lors de la migration du site vers cartes.gouv.fr (redirection
# vers une page de recherche générique). On utilise à la place le miroir
# stable et scriptable de la mission Etalab, qui republie ce même
# référentiel officiel (Code;Libellé) depuis 2018.
RPG_REF_CULTURES_URL = (
    "https://raw.githubusercontent.com/etalab/api-rpg/master/codes/CULTURE.csv"
)

SEASON_START = "2023-11-01"
SEASON_END = "2024-08-31"

# --------------------------------------------------------------------------
# Sentinel-2 (Microsoft Planetary Computer, collection sentinel-2-l2a)
# --------------------------------------------------------------------------
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_COLLECTION = "sentinel-2-l2a"
BANDS = ["B04", "B08", "SCL"]
MAX_CLOUD_COVER_PCT = 60  # filtre au niveau métadonnées de la scène

# Classes SCL à conserver (végétation, sol nu) ; on exclut nuages, ombres de
# nuages, cirrus, neige/glace, eau et pixels saturés/défectueux/no-data —
# une parcelle agricole n'est ni de l'eau ni un nuage.
SCL_KEEP_CLASSES = [4, 5]  # 4 = vegetation, 5 = not vegetated (bare soil)

# Composites temporels : médiane NDVI par fenêtre de 15 jours sur la saison.
COMPOSITE_FREQ_DAYS = 15

# Depuis la baseline de traitement ESA 04.00 (bascule le 2022-01-25), les DN
# des bandes L2A intègrent un décalage additif BOA_ADD_OFFSET = -1000 (pour
# rester en entiers non signés malgré des réflectances pouvant être proches
# de 0). Notre saison (nov. 2023 -> août 2024) est entièrement postérieure à
# cette bascule : toutes les scènes utilisées portent ce décalage. Il n'affecte
# pas le numérateur du NDVI (il s'annule) mais fausse le dénominateur si on
# ne le corrige pas -> à ajouter aux DN bruts avant de calculer le NDVI.
S2_BOA_OFFSET = -1000

# --------------------------------------------------------------------------
# Extraction des signatures parcellaires
# --------------------------------------------------------------------------
PARCEL_NEGATIVE_BUFFER_M = -15  # évite les pixels de bordure/mélange
MIN_PARCEL_AREA_HA = 0.5

# --------------------------------------------------------------------------
# Sélection des classes de cultures
# --------------------------------------------------------------------------
# On ne fixe pas de codes RPG en dur ici : la liste des classes majoritaires
# est déterminée à l'exécution à partir des parcelles réellement présentes
# dans la zone (cf. extract_parcel_signatures.py), les libellés venant de la
# table des codes cultures PAC (RPG_REF_CULTURES_URL ci-dessus).
N_TARGET_CLASSES = 8
MIN_PARCELS_PER_CLASS = 50

# Codes RPG correspondant à des catégories administratives (gel, bordures,
# surfaces non exploitées...) plutôt qu'à une vraie culture avec une
# signature NDVI saisonnière propre. Exclus des classes cibles quel que
# soit leur effectif dans la zone.
NON_CROP_CODE_CULTU = {"SNE", "BOR", "JAC"}  # JAC = jachère (code_group 11, Gel)

# --------------------------------------------------------------------------
# Classification et validation spatiale
# --------------------------------------------------------------------------
# Deux parcelles voisines partagent quasi le même sol, le même microclimat et
# souvent le même exploitant : un split aléatoire les sépare en train/test et
# gonfle artificiellement le score (fuite spatiale). On bloque plutôt la zone
# en cellules géographiques, chaque cellule restant entièrement du côté train
# OU test/fold : deux parcelles voisines sont alors forcément dans le même lot.
SPATIAL_BLOCK_SIZE_M = 2_000  # ~5x5 blocs sur la zone 10x10 km
N_CV_FOLDS = 5
TEST_SIZE = 0.25

MODEL_OUTPUT_PATH = DATA_INTERIM_DIR / "trained_model.joblib"

# --------------------------------------------------------------------------
# Reproductibilité
# --------------------------------------------------------------------------
RANDOM_SEED = 42
