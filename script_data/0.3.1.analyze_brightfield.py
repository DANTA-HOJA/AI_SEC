import sys
from pathlib import Path

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.lif.brightfieldanalyzer import BrightfieldAnalyzer
from modules.shared.utils import get_repo_root
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

brightfield_analyzer = BrightfieldAnalyzer()
brightfield_analyzer.run("0.3.1.analyze_brightfield.toml")