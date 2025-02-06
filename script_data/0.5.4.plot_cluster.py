import sys
from pathlib import Path

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.plot.clustering.surfaceareakmeansplotter import SurfaceAreaKMeansPlotter
from modules.shared.utils import get_repo_root

import matplotlib; matplotlib.use("agg")
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

sa_kmeans_plotter = SurfaceAreaKMeansPlotter()
sa_kmeans_plotter.run("0.5.cluster_data.toml")