import os
import re
import sys
import traceback
from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KernelDensity

from ...data.clustering.utils import log
from ...data.processeddatainstance import ProcessedDataInstance
from ...shared.baseobject import BaseObject
from ...shared.config import load_config
from ...shared.utils import create_new_dir

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# -----------------------------------------------------------------------------/


class SurfaceAreaKMeansPlotter(BaseObject):

    def __init__(self, processed_data_instance:ProcessedDataInstance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("SurfaceArea KMeans Plotter")
        
        if processed_data_instance:
            self._processed_di = processed_data_instance
        else:
            self._processed_di = ProcessedDataInstance()
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.kde_kwargs = {"bandwidth": 0.01178167723136119, "kernel": 'gaussian'}
        
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super()._set_attrs(config)
        self._processed_di.parse_config(config)
        
        self._set_clustered_file_attrs()
        self.clustered_df: pd.DataFrame = \
            pd.read_csv(self.clustered_file, encoding='utf_8_sig')
        
        self._set_surface_area()
        self._set_clusters_max_area_dict()
        self._set_plot_attrs()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """ Set below attributes
            - `self.batch_names`: List[str]
            - `self.random_seed`: int
            - `self.n_class`: int
            - `self.labels`: List[str]
            - `self.cluster_with_log_scale`: bool
            - `self.log_base`: int
            - `self.x_axis_log_scale`: bool
            - `self.old_classdiv_xlsx_list`: List[str]
        """
        """ [batch_info] """
        self.batch_names: List[str] = self.config["batch_info"]["names"]
        
        """ [cluster] """
        self.random_seed: int = self.config["cluster"]["random_seed"]
        self.n_class: int = self.config["cluster"]["n_class"]
        self.labels: List[str] = self.config["cluster"]["labels"]
        self.cluster_with_log_scale: bool = self.config["cluster"]["cluster_with_log_scale"]
        
        """ [log_scale] """
        self.log_base: int = self.config["log_scale"]["base"]
        self.x_axis_log_scale: bool = self.config["log_scale"]["x_axis_log_scale"]
        
        """ [old_classdiv_xlsx] """
        self.old_classdiv_xlsx_list: List[str] = self.config["old_classdiv_xlsx"]["abs_paths"]
        # ---------------------------------------------------------------------/


    def _set_clustered_file_attrs(self):
        """ Set below attributes
            - self.clustered_file
            - self.clustered_desc
            - self.dst_root
        """
        def gen_clustered_desc() -> str:
            if self.cluster_with_log_scale:
                name = f"SURF{self.n_class}C_KMeansLOG{self.log_base}_RND{self.random_seed}"
            else:
                name = f"SURF{self.n_class}C_KMeansORIG_RND{self.random_seed}"
            return name
            # -----------------------------------------------------------------
        
        desc: str = gen_clustered_desc()
        try:
            self.clustered_file: Path = self._processed_di.clustered_files_dict[desc]
        except KeyError:
            traceback.print_exc()
            print(f"{Fore.RED}{Back.BLACK} Can't find `{{{desc}}}_datasplit.csv`, "
                  f"please run `0.5.3.cluster_data.py` to create it. {Style.RESET_ALL}\n")
            sys.exit()
        
        self.clustered_desc = desc
        self.dst_root: Path = Path(os.path.dirname(self.clustered_file))
        # ---------------------------------------------------------------------/


    def _set_surface_area(self):
        """
        """
        if self.x_axis_log_scale:
            self.clustered_df["Trunk surface area, SA (um2)"] = \
                self.clustered_df["Trunk surface area, SA (um2)"].apply(lambda x: log(self.log_base, x))
        
        self.surface_area: np.ndarray = \
            self.clustered_df["Trunk surface area, SA (um2)"].to_numpy()
        # ---------------------------------------------------------------------/


    def _set_clusters_max_area_dict(self):
        """
        """
        self.clusters_max_area_dict: Dict[str, float] = {}
        
        for label in self.labels:
            df = self.clustered_df[self.clustered_df["class"] == label]
            max_area = df["Trunk surface area, SA (um2)"].max()
            self.clusters_max_area_dict[label] = max_area
        # ---------------------------------------------------------------------/


    def _set_plot_attrs(self):
        """
        """
        self.fig, self.ax = \
            plt.subplots(1, 1, figsize=(16, 16), dpi=200) # figure 內的 element 都是用比例決定位置的
                                                          #  一旦決定 figsize 之後如果要調整圖片"長寬比"最好只調整一邊
                                                          #  調整"整張圖片大小" -> dpi
        self.divider = make_axes_locatable(self.ax)
        self.fig.subplots_adjust(top=0.9)
        
        # scatter attrs ( scatter y 軸起始位置 )
        self.scatter_init_pos: float = 0.0
        if self.x_axis_log_scale:
            self.scatter_init_pos = -0.175
        else: self.scatter_init_pos = -8e-8
        self.scatter_n_gap: int = 0 # 控制 scatter 之間的間距: self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)
        
        # digits ( 圖上 "浮點數" 的小數位數 )
        self.digits: int = 0
        if self.x_axis_log_scale:
            self.digits = 8
        else: self.digits = 2

        # text_path_effect ( 文字陰影 )
        self.text_path_effect = path_effects.withSimplePatchShadow(
                                    offset=(0.5, -0.5), linewidth=1, foreground='black')
        
        self.ax_x_lim: Tuple[float, float] # 記憶 self.ax.get_xlim()
        
        # color maps
        self.fish_class_cmap: str = "viridis"
        self.fish_batch_cmap: str = "Paired"
        self.fish_day_cmap: str = "Dark2"
        
        # Legend and colorbar
        self.fish_class_legend: Legend
        self.fish_batch_legend: Legend
        self.fish_day_legend: Legend
        self.colorbar_n_gap: int = 0 # 控制 colorbar 之間的間距
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        hist = self._plot_hist()
        if self.x_axis_log_scale: self._plot_kde(hist)
        
        # cluster ( class )
        self._plot_cluster_distribution()
        self._plot_cluster_center()
        self._plot_cluster_boundary()
        self._plot_cluster_count()
        self.ax_x_lim = self.ax.get_xlim()
        
        # batch
        self._plot_dividing_line()
        self._plot_batch_distribution()
        
        # day
        self._plot_dividing_line()
        self._plot_day_distribution()
        
        # update Legends and colorbars
        self.ax.get_legend().remove()
        self.ax.add_artist(self.fish_class_legend)
        self.ax.add_artist(self.fish_batch_legend)
        self._add_colorbar("day")
        
        self._save_fig()
        self._plot_old_classdiv_xlsx()
        
        plt.close(self.fig)
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _plot_hist(self):
        """
        """
        hist = self.ax.hist(self.surface_area, bins=100, density=True, alpha=0.7)
        density, bins, patches = hist
        widths = bins[1:] - bins[:-1]
        self._cli_out.write(f"hist_accum_p = {(density * widths).sum()}")
        
        return hist
        # ---------------------------------------------------------------------/


    def _plot_kde(self, hist):
        """ instantiate and fit the KDE model
        """
        _, bins, _ = hist
        
        kde = KernelDensity(**self.kde_kwargs)
        kde.fit(self.surface_area[:, None]) # reshape to 2-D array: (100) -> (100, )

        # score_samples returns the log of the probability density
        logprob = kde.score_samples(bins[:, None])

        self.ax.fill_between(bins, np.exp(logprob), alpha=0.5, color="orange")
        
        self.surface_area = self.surface_area.squeeze() # revert to 1-D array
        # ---------------------------------------------------------------------/


    def _plot_marks_scatter(self, attr_prefix:str,
                           mark_style:str, separately:bool, legend_name:str,
                           legend_loc:str, legend_bbox_to_anchor:Tuple[float, float]):
        """
        """
        df_col_name = attr_prefix.replace("fish_", "")
        ordered_list = self._get_ordered_list(df_col_name)
        
        colormap = matplotlib.colormaps[getattr(self, f"{attr_prefix}_cmap")]
        colors = colormap(np.linspace(0, 1, len(ordered_list)))
        
        scatter_list:list = []
        for i, text in enumerate(ordered_list):
            
            df = self.clustered_df[(self.clustered_df[df_col_name] == text)]
            surface_area_list = list(df["Trunk surface area, SA (um2)"])
        
            scatter = self.ax.scatter(
                surface_area_list, np.full_like(surface_area_list, self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
                edgecolor = 'none',   # 無邊框
                # alpha = 0.5         # 不透明度
                color = colors[i],
                label = text,
                marker = mark_style
            )
            scatter_list.append(scatter)
            
            if separately: self.scatter_n_gap += 1
            else:
                # show counter info ( always start from `df_col_name`)
                df_col_list = ["class", "batch", "day"]
                output_string = ""
                for j, value in enumerate(df_col_list):
                    if value == df_col_name:
                        key, value = list(Counter(df[df_col_name]).items())[0]
                        output_string += f"{legend_name}_{key}: {value}"
                        df_col_list.pop(j) # 移除顯示過的
                        for value in df_col_list:
                            cnt = Counter(df[value])
                            temp_list = self._get_ordered_list(value)
                            temp_dict = { item: cnt[item] for item in temp_list }
                            output_string += f", {temp_dict}"
                            j += 1
                        self._cli_out.write(output_string)
                        break
        
        # create the legend set of scatters
        setattr(self, f"{attr_prefix}_legend", self.ax.legend(handles=scatter_list,
                                                              title=legend_name,
                                                              loc=legend_loc,
                                                              bbox_to_anchor=legend_bbox_to_anchor))
        # ---------------------------------------------------------------------/


    def _get_ordered_list(self, df_col_name:str) -> list:
        """
        """
        if df_col_name == "class":
            ordered_list = deepcopy(self.labels)
        elif df_col_name == "batch":
            ordered_list = sorted(Counter(self.clustered_df["batch"]).keys(),
                                  key=lambda x: int(x.split('i')[-1]))
        elif df_col_name == "day":
            ordered_list = sorted(Counter(self.clustered_df["day"]).keys())
        else:
            raise NotImplementedError
        
        return ordered_list
        # ---------------------------------------------------------------------/


    def _plot_cluster_distribution(self):
        """
        """
        """ area ( black ticks ) """
        self.ax.plot(self.surface_area, np.full_like(self.surface_area, self.scatter_init_pos),
                     '|k', markeredgewidth=1)
        
        """ cluster ( circle ) """
        kwargs = {
            "attr_prefix"           : "fish_class", # 用來存取 self.[obj_prefix]_cmap 之類的變數
            "mark_style"            : 'o',
            "separately"            : False, # 分開畫
            "legend_name"           : "cluster",
            "legend_loc"            : 'upper right',
            "legend_bbox_to_anchor" : (1.0, 0.99) # 0 是靠左，1 是靠右
        }
        self._plot_marks_scatter(**kwargs)
        # ---------------------------------------------------------------------/


    def _plot_cluster_center(self):
        """
        """
        file = self.dst_root.joinpath("kmeans_centers.toml")
        if not file.exists():
            self._cli_out.write(f"Can't find file: '{file}', "
                                "'kmeans_centers' will not plot")
            return
        
        toml_file: dict = load_config(file)
        kmeans_centers = np.array(list(toml_file.values()))
        if self.x_axis_log_scale:
            kmeans_centers = log(self.log_base, kmeans_centers)
        
        self.ax.scatter(kmeans_centers, np.full_like(kmeans_centers, self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
                        marker="x", s=50, color="black")
        
        # update `self.scatter_n_gap`
        self.scatter_n_gap += 1
        # ---------------------------------------------------------------------/


    def _plot_cluster_boundary(self):
        """
        """
        # min_value line
        min_boundary = self.surface_area.min()
        self.ax.axvline(x=min_boundary, color='k', linestyle='--')
        self.ax.text(min_boundary, 0.95, f'  x={min_boundary:.{self.digits}f}',
                         transform=self.ax.get_xaxis_transform(), ha='left',
                         color='black', path_effects=[self.text_path_effect])
        
        # cluster_max_value lines
        text_y_pos = {0: 0.92, 1: 0.95} # 文字上下交錯
        for i, boundary in enumerate(self.clusters_max_area_dict.values()):
            self.ax.axvline(x=boundary, color='k', linestyle='--')
            self.ax.text(boundary, text_y_pos[(i%2)], f'x={boundary:.{self.digits}f}  ',
                         transform=self.ax.get_xaxis_transform(), ha='right',
                         color='black', path_effects=[self.text_path_effect])
        # ---------------------------------------------------------------------/


    def _plot_cluster_count(self):
        """
        """
        value_list = list(self.clusters_max_area_dict.values()) # [100, 200, 300]
        value_list.insert(0, self.surface_area.min()) # [0, 100, 200, 300]
        
        clusters_count = Counter(self.clustered_df["class"])
        
        for i, label in enumerate(self.labels):
            text_center = (value_list[i+1] + value_list[i])/2
            text = self.ax.text(text_center, 0.8, f'{label}={clusters_count[label]}',
                                transform=self.ax.get_xaxis_transform(), ha='center',
                                fontsize=16, color='#FFFFF2', path_effects=[self.text_path_effect])
            text.set_bbox(dict(boxstyle="round", pad=0.8, facecolor='#EE7785', alpha=0.7, edgecolor='none',
                               path_effects=[path_effects.withSimplePatchShadow(offset=(2, -2), foreground='black')]))
        # ---------------------------------------------------------------------/


    def _plot_dividing_line(self):
        """
        """
        self.ax.hlines(self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83),
                       self.ax_x_lim[0], self.ax_x_lim[1],
                       color="dodgerblue", linestyles='dashed')
        self.scatter_n_gap += 1
        # ---------------------------------------------------------------------/


    def _plot_batch_distribution(self):
        """
        """
        for flag in [False, True]:
            kwargs = {
                "attr_prefix"           : "fish_batch", # 用來存取 self.[obj_prefix]_cmap 之類的變數
                "mark_style"            : 'o',
                "separately"            : flag, # 分開畫
                "legend_name"           : "batch",
                "legend_loc"            : 'upper left',
                "legend_bbox_to_anchor" : (0.0, 0.99) # 0 是靠左，1 是靠右
            }
            self._plot_marks_scatter(**kwargs)
            
            if not flag: self.scatter_n_gap += 1 # 畫完重疊的 scatter 後要加一次間距
        # ---------------------------------------------------------------------/


    def _plot_day_distribution(self):
        """
        """
        # scatter
        for flag in [False, True]:
            kwargs = {
                "attr_prefix"           : "fish_day", # 用來存取 self.[obj_prefix]_cmap 之類的變數
                "mark_style"            : 's',
                "separately"            : flag, # 分開畫
                "legend_name"           : "day",
                "legend_loc"            : 'upper right',
                "legend_bbox_to_anchor" : (1.0, 0.88) # 0 是靠左，1 是靠右
            }
            self._plot_marks_scatter(**kwargs)
            
            if not flag: self.scatter_n_gap += 1 # 畫完重疊的 scatter 後要加一次間距
        # ---------------------------------------------------------------------/


    def _add_colorbar(self, df_col_name:str):
        """
        """
        enum_dict = {value: i for i, value in enumerate(self._get_ordered_list(df_col_name))}
        target_list = [ enum_dict[i] for i in self.clustered_df[df_col_name] ]
        
        
        cax = self.divider.append_axes("right", "2%", pad=0.3+self.colorbar_n_gap*0.2) # "右側" 加新的軸
        self.colorbar_n_gap += 1
        
        mappable = cm.ScalarMappable(cmap=getattr(self, f"fish_{df_col_name}_cmap"))
        mappable.set_array(target_list) # 會自動統計 items 並排序
        cbar = self.fig.colorbar(mappable, cax=cax) # create a `color_bar`
        cbar.set_ticks(list(enum_dict.values()), labels=list(enum_dict.keys()))
        cbar.ax.set_xlabel(df_col_name, labelpad=10) # 設置 `color_bar` 的標籤
        # ---------------------------------------------------------------------/


    def _save_fig(self):
        """
        """
        # set title
        dataset_inum = self._processed_di.instance_name.split("_")[-1]
        self.fig_title: str = f"{dataset_inum}, {self.clustered_desc}{', KDE' if self.x_axis_log_scale else ''}"
        self.fig.suptitle(self.fig_title, size=20)
        
        # save figure
        self.fig_file_name = f"{{{self.clustered_desc}}}{'_kde' if self.x_axis_log_scale else ''}"
        self.fig.savefig(self.dst_root.joinpath(f"{self.fig_file_name}.png"))
        self.fig.savefig(self.dst_root.joinpath(f"{self.fig_file_name}.svg"))
        # ---------------------------------------------------------------------/


    def _plot_old_classdiv_xlsx(self):
        """
        """
        for old_classdiv_xlsx_file in self.old_classdiv_xlsx_list:
            
            try:
                # check item
                if old_classdiv_xlsx_file == "":
                    raise ValueError(f"`old_classdiv_xlsx_file` can't be an empty string")
                
                # check exists
                old_classdiv_xlsx_file = Path(old_classdiv_xlsx_file).resolve()
                if not old_classdiv_xlsx_file.exists():
                    raise FileNotFoundError(f"Can't find file: '{old_classdiv_xlsx_file}'")

                # get vars
                old_classdiv_strategy, compare_dir, \
                    old_classdiv_info_dict = self._get_old_xlsx_attrs(old_classdiv_xlsx_file)
                
            except (FileNotFoundError, ValueError, AssertionError) as e:
                self._cli_out.write(f"{Fore.RED}{Back.BLACK}\n\n{e}{Style.RESET_ALL}")
                self._cli_out.write(f"{Fore.YELLOW}{Back.BLACK}Compare figure will not generate{Style.RESET_ALL}")
                continue
            
            fig = self._plot_old_classdiv_boundary(self.fig, old_classdiv_info_dict)
            self._save_fig_with_old_classdiv(fig, old_classdiv_strategy, compare_dir)
        # ---------------------------------------------------------------------/


    def _get_old_xlsx_attrs(self, old_classdiv_xlsx_file:Path):
        """
        """
        old_classdiv_xlsx_name: str = str(old_classdiv_xlsx_file).split(os.sep)[-1] # '{3CLS_SURF_050STDEV}_data.xlsx'
        name_split: List[str] = re.split("{|_|}", old_classdiv_xlsx_name) # ['', '3CLS', 'SURF', '050STDEV', '', 'data.xlsx']
        
        """ Check old xlsx class """
        # 'SURF10C' -> int(10)
        match = re.search(r'\d+', name_split[1]) # 匹配一個或多個連續數字
        if match: number = int(match.group()) # 如果找到匹配，取出匹配的結果
        else: raise ValueError(f"File name: '{old_classdiv_xlsx_name}' is not correct format.")
        assert number == self.n_class, (f"`{old_classdiv_xlsx_name}` not match to `n_class`, "
                                        f"expect {self.n_class}, but got {number}, "
                                        f"file: '{old_classdiv_xlsx_file}'")
        
        """ Get `old_classdiv_strategy` """
        # '050STDEV' -> '0.5_STDEV'
        stdev = int(name_split[3].replace("STDEV", ""))/100
        old_classdiv_strategy: str = f"{stdev}_STDEV"

        """ Create directory """
        compare_dir = self.dst_root.joinpath("compare_clustering", "KMeans_comp_STDEV")
        if self.x_axis_log_scale:
            compare_dir = compare_dir.joinpath(f"x-axis in LOG{self.log_base} scale")
        else: compare_dir = compare_dir.joinpath("x-axis in ORIG scale")
        create_new_dir(compare_dir)
        
        """ Set `old_classdiv_info_dict` """
        old_classdiv_xlsx_df = pd.read_excel(old_classdiv_xlsx_file, engine = 'openpyxl')
        old_classdiv_info_dict = {}
        if self.x_axis_log_scale:
            old_classdiv_info_dict['L_std_value'] = log(self.log_base, old_classdiv_xlsx_df["L_1s"][0])
            old_classdiv_info_dict['avg_value']   = log(self.log_base, old_classdiv_xlsx_df["average"][0])
            old_classdiv_info_dict['R_std_value'] = log(self.log_base, old_classdiv_xlsx_df["R_1s"][0])
        else: 
            old_classdiv_info_dict['L_std_value'] = old_classdiv_xlsx_df["L_1s"][0]
            old_classdiv_info_dict['avg_value']   = old_classdiv_xlsx_df["average"][0]
            old_classdiv_info_dict['R_std_value'] = old_classdiv_xlsx_df["R_1s"][0]
        
        return old_classdiv_strategy, compare_dir, old_classdiv_info_dict
        # ---------------------------------------------------------------------/


    def _plot_old_classdiv_boundary(self, figure:Figure,
                                   old_classdiv_info_dict:Dict[str, float]):
        """
        """
        fig: Figure = deepcopy(figure)
        ax = fig.axes[0]
        
        for i, (key, value) in enumerate(old_classdiv_info_dict.items()):
            ax.axvline(x=value, color='r', linestyle='--', alpha=0.7)
            ax.text(value, 0.666, f'  {key:{self.digits}}:\n  {value:.{self.digits}f}',
                    transform=ax.get_xaxis_transform(), ha='left',
                    color='red', path_effects=[self.text_path_effect], alpha=0.7)
        
        return fig
        # ---------------------------------------------------------------------/


    def _save_fig_with_old_classdiv(self, figure:Figure,
                                   old_classdiv_strategy:str, save_dir:Path):
        """
        """
        self.fig.savefig(save_dir.joinpath(f"{self.fig_file_name}.png"))
             
        old_classdiv_fig_title = f"{self.fig_title}, {old_classdiv_strategy}"
        figure.suptitle(old_classdiv_fig_title, size=20)
        
        old_classdiv_fig_save_name = f"{self.fig_file_name}_{old_classdiv_strategy}"
        save_file = save_dir.joinpath(f"{old_classdiv_fig_save_name}.png")
        figure.savefig(save_file)
        self._cli_out.write(f"Compare figure: '{save_file}'")
        # ---------------------------------------------------------------------/