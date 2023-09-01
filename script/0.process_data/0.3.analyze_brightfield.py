import sys
from pathlib import Path

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.brightfieldanalyzer import BrightfieldAnalyzer
from modules.shared.utils import get_repo_root

""" Detect Repository """
repo_root = get_repo_root(display_on_CLI=True)


brightfield_analyzer = BrightfieldAnalyzer()
brightfield_analyzer.run()