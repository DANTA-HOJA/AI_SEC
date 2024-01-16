import sys
import traceback

from colorama import Back, Fore, Style
from rich import print
from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty, pprint
from rich.text import Text
from rich.traceback import install

install(max_frames=1)

console = Console()

panel_group = Group(
    Panel("Hello", style="on blue"),
    Panel("World", style="on red"),
)

# raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find `dataset_{2022}.csv`, "
#                         f"please run `0.5.2.split_dataset.py` to create it. {Style.RESET_ALL}\n")