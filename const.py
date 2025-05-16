from pathlib import Path

# Get the directory where const.py is located (project root)
PROJECT_ROOT = Path(__file__).parent

# Data paths as absolute paths
DATA_DIR = PROJECT_ROOT / "data"
BONE_DATA_PATH = DATA_DIR / "bone.data"
SPLINE_SAMPLE_DATA_PATH = DATA_DIR / "spline_sample.csv"

RANDOM_SEED = 94742
