import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Union, Tuple
from collections import OrderedDict, Counter
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from colorama import Fore, Back, Style

from .. import dname
from ..processeddatainstance import ProcessedDataInstance
from ...shared.clioutput import CLIOutput
from ...shared.config import load_config
from ...shared.utils import create_new_dir



class SurfaceAreaKMeansCluster():
    
    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        self.processed_data_instance = ProcessedDataInstance()
        
        """ CLI output """
        self._cli_out = CLIOutput(display_on_CLI, logger_name="SurfaceArea KMeans Cluster")
    
    
    
    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.processed_data_instance.set_attrs(config_file)
        self._set_config_attrs()
        
        self._set_orig_xlsx_attrs()
        
        self.bf_dnames:List[str] = list(self.orig_xlsx_df["Brightfield"])
        self.surface_area = self.orig_xlsx_df["Trunk surface area, SA (um2)"].to_numpy()[:, None] # reshape: (100) -> (100, )
        self.batch_idx_list:List[int] = self._set_batch_idx_list() # data 對應的 batch index
        self.day_list:List[int] = self._set_day_list() # data 對應的 day (dpf)
        
        self.kmeans = KMeans(n_clusters = self.n_class, random_state=self.random_seed)
        self.kmeans_centers = None # assigned after `run_kmeans()`
        self.y_kmeans = None # assigned after `run_kmeans()`
        self.cidx_max_area_dict = None # assigned after `_set_cidx_max_area_dict()`
        self.cidx2clabel = None # assigned after `_set_cidx2clabel()`
        
        self._set_clustered_xlsx_attrs()
        # -------------------------------------------------------------------------------------
    
    
    
    def _set_config_attrs(self, config_file:Union[str, Path]):
        """ Set below attributes
            - `self.batch_id_interval`: List[int]
            - `self.batch_idx2str`: Dict[int, str]
            - `self.random_seed`: int
            - `self.n_class`: int
            - `self.labels`: List[str]
            - `self.cluster_with_log_scale`: bool
            - `self.log_base`: int
            - `self.x_axis_log_scale`: bool
        """
        config = load_config(config_file, cli_out=self._cli_out)
        self.config = config
        
        """ [batch_info] """
        self.batch_id_interval: List[int] = config["batch_info"]["id_interval"]
        self.batch_idx2str: Dict[int, str] = {i: name for i, name in enumerate(config["batch_info"]["names"])}
        
        """ [cluster] """
        self.random_seed: int = config["cluster"]["random_seed"]
        self.n_class: int = config["cluster"]["n_class"]
        self.labels: List[str] = config["cluster"]["labels"]
        self.cluster_with_log_scale: bool = config["cluster"]["cluster_with_log_scale"]
        
        """ [log_scale] """
        self.log_base: int = config["log_scale"]["base"]
        self.x_axis_log_scale: bool = config["log_scale"]["x_axis_log_scale"]
        # -------------------------------------------------------------------------------------
    
    
    
    def _set_orig_xlsx_attrs(self):
        """ Set below attributes
            - `self.orig_xlsx_path`: Path
            - `self.orig_xlsx_df`: pd.DataFrame
            - `self.orig_xlsx_path_split`: List[str]
        """
        def set_attr1() -> Path:
            if self.processed_data_instance.data_xlsx_path:
                return self.processed_data_instance.data_xlsx_path
            else:
                raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find `data.xlsx` please run `0.5.create_data_xlsx.py` to create it. {Style.RESET_ALL}\n")
        
        self.orig_xlsx_path: Path = set_attr1()
        self.orig_xlsx_df: pd.DataFrame = pd.read_excel(self.orig_xlsx_path, engine = 'openpyxl')
        self.orig_xlsx_path_split: List[str] = str(self.orig_xlsx_path).split(os.sep)
        # -------------------------------------------------------------------------------------
    
    
    
    def _set_clustered_xlsx_attrs(self):
        """ Set below attributes
            - `self.clustered_xlsx_dir`: Path
            - `self.clustered_xlsx_name`: str
            - `self.clustered_xlsx_path`: Path
            - `self.clustered_xlsx_df` # assigned after `gen_clustered_xlsx_df()`
        """
        def set_attr1() -> Path:
            if self.processed_data_instance.clustered_xlsx_dir:
                path = self.processed_data_instance.clustered_xlsx_dir
            else:
                path = self.processed_data_instance.instance_root.joinpath("Clustered_xlsx")
                create_new_dir(path)
            return path
        
        def set_attr2() -> str:
            if self.cluster_with_log_scale:
                name = f"{self.n_class}CLS_SURF_KMeansLOG{self.log_base}_RND{self.random_seed}"
            else:
                name = f"{self.n_class}CLS_SURF_KMeansORIG_RND{self.random_seed}"
            return name
            
        self.clustered_xlsx_dir: Path = set_attr1()
        self.clustered_xlsx_name: str = set_attr2()
        self.clustered_xlsx_path: Path = self.clustered_xlsx_dir.joinpath(f"{{{self.clustered_xlsx_name}}}_data.xlsx")
        self.clustered_xlsx_df = None # assigned after `gen_clustered_xlsx_df()`
        # -------------------------------------------------------------------------------------
    
    
    
    def _set_batch_idx_list(self):
        """
        """
        batch_idx_list: List[int] = deepcopy(self.bf_dnames)
        
        for i, bf_dname in enumerate(self.bf_dnames):
            fish_id = dname.get_dname_sortinfo(bf_dname)[0]
            for j in range(len(self.batch_id_interval)-1):
                if (fish_id > self.batch_id_interval[j]) and (fish_id <= self.batch_id_interval[j+1]):
                    batch_idx_list[i] = j
                    break
        
        return batch_idx_list
        # -------------------------------------------------------------------------------------
    
    
    
    def _set_day_list(self):
        """
        """
        day_list: List[int] = deepcopy(self.bf_dnames)
        
        for i, bf_dname in enumerate(self.bf_dnames):
            fish_day = int(re.split(" |_|-", bf_dname)[3].replace("dpf", ""))
            day_list[i] = fish_day
        
        return day_list
        # -------------------------------------------------------------------------------------
    
    
    
    def log(self, base, x):
        """
        """
        return np.log(x) / np.log(base)
        # -------------------------------------------------------------------------------------
    
    
    
    def run_kmeans(self):
        """
        """
        if self.cluster_with_log_scale:
            self.surface_area = self.log(self.log_base, self.surface_area)
        
        self.kmeans.fit(self.surface_area)
        self.y_kmeans = self.kmeans.predict(self.surface_area) # 產生分群結果
        self.kmeans_centers = self.kmeans.cluster_centers_ # 取得群心
        print(f'kmeans_centers {type(self.kmeans_centers)}: \n{self.kmeans_centers}\n')
        
        if self.cluster_with_log_scale and (not self.x_axis_log_scale):
            """ 只有分類時取 LOG, 畫圖不要取 LOG --> data 還原為原始刻度（量級）"""
            self.surface_area   = self.log_base ** self.surface_area
            self.kmeans_centers = self.log_base ** self.kmeans_centers
        
        if (not self.cluster_with_log_scale) and self.x_axis_log_scale:
            """ 分類不要取 LOG, 畫圖要取 LOG --> 將 data 刻度（量級）取 LOG """
            self.surface_area   = self.log(self.log_base, self.surface_area)
            self.kmeans_centers = self.log(self.log_base, self.kmeans_centers)
        # -------------------------------------------------------------------------------------
    
    
    
    def _set_cidx_max_area_dict(self):
        """ execution dependency:
            - `self.n_class`
            - `self.surface_area`
            - `self.y_kmeans`
        """
        cidx_max_area_list = [0]*self.n_class
        for area, cidx in zip(self.surface_area.squeeze(), self.y_kmeans):
            if area > cidx_max_area_list[cidx]:
                cidx_max_area_list[cidx] = area

        self.cidx_max_area_dict = {i: max_area for i, max_area in enumerate(cidx_max_area_list)}
        self.cidx_max_area_dict = OrderedDict(sorted(list(self.cidx_max_area_dict.items()), key=lambda x: x[1]))
        self._cli_out.write(f"self.cidx_max_area_dict : {self.cidx_max_area_dict}")
        # -------------------------------------------------------------------------------------
    
    
    
    def _set_cidx2clabel(self):
        """ execution dependency:
            - `self.cidx_max_area_dict`
        """
        self.cidx2clabel = {cidx: clabel for cidx, clabel in zip(self.cidx_max_area_dict.keys(), self.labels) }
        self._cli_out.write(f"self.cidx2clabel : {self.cidx2clabel}")
        # -------------------------------------------------------------------------------------
    
    
    
    def gen_clustered_xlsx_df(self):
        """ execution dependency:
            - `self.orig_xlsx_df`
            - `self.y_kmeans`
            - `self.cidx2clabel`
            - `self.batch_idx_list`
            - `self.batch_idx2str`
            - `self.day_list`
        """
        """ Add class column """
        clabel_list = [self.cidx2clabel[cidx] for cidx in self.y_kmeans]
        new_col = pd.Series(clabel_list, name="class")
        self.clustered_xlsx_df = pd.concat([self.orig_xlsx_df, new_col], axis=1)
        
        """ Add batch column """
        batch_str_list = [self.batch_idx2str[batch_idx] for batch_idx in self.batch_idx_list]
        new_col = pd.Series(batch_str_list, name="batch")
        self.clustered_xlsx_df = pd.concat([self.clustered_xlsx_df, new_col], axis=1)
        
        """ Add day column """
        new_col = pd.Series(self.day_list, name="day")
        self.clustered_xlsx_df = pd.concat([self.clustered_xlsx_df, new_col], axis=1)
        # -------------------------------------------------------------------------------------
    
    
    
    def save_clustered_xlsx_df(self):
        """ execution dependency:
            - `self.clustered_xlsx_path`
            - `self.clustered_xlsx_df`
        """
        self.clustered_xlsx_df.to_excel(self.clustered_xlsx_path, engine="openpyxl", index=False)
        # -------------------------------------------------------------------------------------
    
    
    
    def run(self):
        """
        """
        self.run_kmeans()
        self._set_cidx_max_area_dict()
        self._set_cidx2clabel()
        self.gen_clustered_xlsx_df()
        self.save_clustered_xlsx_df()
        # -------------------------------------------------------------------------------------
    
    
    
    def __repr__(self):
        """
        """
        return json.dumps(self.config, indent=4)
        # -------------------------------------------------------------------------------------
    
    
    
    
    # def plot_misc_settings(self):
        
    #     if self.x_axis_log_scale: self.scatter_init_pos = -0.175
    #     else: self.scatter_init_pos = -8e-8
        
    #     if self.x_axis_log_scale: self.digits = 8
    #     else: self.digits = 2

    #     self.text_path_effect = path_effects.withSimplePatchShadow(
    #                                 offset=(0.5, -0.5), linewidth=1, foreground='black')
    #     # -------------------------------------------------------------------------------------
    
    
    # def get_current_xlim(self):
    #     self.ax_x_lim = self.ax.get_xlim()
    #     # -------------------------------------------------------------------------------------
    
    
    # def add_colorbar(self, mapping_list:list, name:str, cmap:str,
    #                        ticks:List, ticklabels:Union[List[str], None]=None):
        
    #     cax = self.divider.append_axes("right", "2%", pad=0.3+self.colorbar_n_gap*0.2) # "右側" 加新的軸
    #     self.colorbar_n_gap += 1                                                       # init: self.divider = make_axes_locatable(self.ax)
    #     mappable = cm.ScalarMappable(cmap=cmap)
    #     mappable.set_array(mapping_list)  # 會自動統計 items 並排序
    #     cbar = self.fig.colorbar(mappable, cax=cax) # create a `color_bar`
    #     cbar.ax.set_xlabel(name, labelpad=10)  # 設置 `color_bar` 的標籤
    #     cbar.set_ticks(ticks, labels=ticklabels)
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_hist(self):
    #     hist = self.ax.hist(self.surface_area, bins=100, density=True, alpha=0.7)
    #     density, self.bins, patches = hist
    #     widths = self.bins[1:] - self.bins[:-1]
    #     print(f"accum_p = {(density * widths).sum()}\n")
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_kde(self):
    #     # instantiate and fit the KDE model
    #     kde = KernelDensity(**self.kde_kwargs)
    #     kde.fit(self.surface_area)

    #     # score_samples returns the log of the probability density
    #     logprob = kde.score_samples(self.bins[:, None])

    #     self.ax.fill_between(self.bins, np.exp(logprob), alpha=0.5, color="orange")
    #     # self.ax.plot(self.bins, np.exp(logprob), label='KDE', color="orange") # , linestyle='--'
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_marks_scatter(self, attr_prefix:str, surf2mark_dict:Dict[float, int], mapping_dict:Dict[int, str],
    #                        mark_style:str, separately:bool, show_info:bool, legend_name:str,
    #                        legend_loc:str, legend_bbox_to_anchor:Tuple[float, float]):
        
    #     colormap = matplotlib.colormaps[getattr(self, f"{attr_prefix}_cmap")]
    #     colors = colormap(np.linspace(0, 1, len(mapping_dict)))
        
    #     setattr(self, f"{attr_prefix}_scatter_list", [])
    #     scatter_list = getattr(self, f"{attr_prefix}_scatter_list") # init as a list
    #     for i, mapping_key in enumerate(mapping_dict.keys()):
            
    #         surface_area_list = []
    #         mark_list = []
    #         for area, mark in surf2mark_dict.items():
    #             if mark == mapping_key:
    #                 surface_area_list.append(area)
    #                 mark_list.append(mark)
        
    #         scatter = self.ax.scatter(
    #             surface_area_list, np.full_like(surface_area_list, self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
    #             edgecolor = 'none',   # 無邊框
    #             # alpha = 0.5         # 不透明度
    #             color=colors[i],
    #             label=mapping_dict[mapping_key],
    #             marker=mark_style
    #         )
    #         scatter_list.append(scatter)
            
    #         if separately: self.scatter_n_gap += 1
    #         if show_info: print(f"{legend_name}_{mapping_key}: '{mapping_dict[mapping_key]}', {len(mark_list)}")
        
    #     # collect and create the legend set of scatters
    #     setattr(self, f"{attr_prefix}_legend", self.ax.legend(handles=scatter_list, 
    #                                                           title=legend_name, 
    #                                                           loc=legend_loc, 
    #                                                           bbox_to_anchor=legend_bbox_to_anchor))
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_cluster_distribution(self):
    #     # black_ticks
    #     self.ax.plot(self.surface_area, np.full_like(self.surface_area, self.scatter_init_pos), 
    #                  '|k', markeredgewidth=1)
    #     # cluster
    #     kwargs = {
    #         "attr_prefix"           : "clusters", # 用來存取 self.[obj_prefix]_scatter_list 之類的變數
    #         "surf2mark_dict"        : self.bfname2cidx_dict,
    #         "mapping_dict"          : self.cidx2clabel,
    #         "mark_style"            : 'o',
    #         "separately"            : False, # 分開畫
    #         "show_info"             : True,
    #         "legend_name"           : "cluster",
    #         "legend_loc"            : 'upper right',
    #         "legend_bbox_to_anchor" : (1.0, 0.99) # 0 是靠左，1 是靠右
    #     }
    #     self.plot_marks_scatter(**kwargs)
    #     # cluster_center
    #     self.ax.scatter(self.kmeans_centers.reshape(-1), 
    #                     np.full_like(self.kmeans_centers.reshape(-1), self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
    #                     marker="x", s=50, color="black")
    #     self.scatter_n_gap += 1
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_batch_mark(self):
        
    #     batch_mark_dict = { area: batch_mark for area, batch_mark in zip(self.surface_area.squeeze(), self.batch_idx_list) }

    #     # dividing line
    #     self.ax.hlines(self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83), 
    #                    self.ax_x_lim[0], self.ax_x_lim[1], 
    #                    color="dodgerblue", linestyles='dashed')
    #     self.scatter_n_gap += 1
        
    #     # scatter
    #     for sw in [False, True]:
    #         kwargs = {
    #             "attr_prefix"           : "fish_batch", # 用來存取 self.[obj_prefix]_scatter_list 之類的變數
    #             "surf2mark_dict"        : batch_mark_dict,
    #             "mapping_dict"          : self.batch_idx2str,
    #             "mark_style"            : 'o',
    #             "separately"            : sw, # 分開畫
    #             "show_info"             : sw,
    #             "legend_name"           : "batch",
    #             "legend_loc"            : 'upper left',
    #             "legend_bbox_to_anchor" : (0.0, 0.99) # 0 是靠左，1 是靠右
    #         }
    #         self.plot_marks_scatter(**kwargs)
            
    #         if not sw: self.scatter_n_gap += 1 # 畫完重疊的 scatter 後要加一次間距
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_day_mark(self):
        
    #     self.fish_day_cnt = Counter(self.day_list)
    #     self.fish_day_cnt = OrderedDict(sorted(self.fish_day_cnt.items(), key=lambda x: x[0])) # sort by day
    #     day_mark_dict = { area: day_mark for area, day_mark in zip(self.surface_area.squeeze(), self.day_list) }
    #     fish_day_mark2str = { key: str(key) for key in self.fish_day_cnt.keys()}
        
    #     # dividing line
    #     self.ax.hlines(self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83), 
    #                    self.ax_x_lim[0], self.ax_x_lim[1], 
    #                    color="dodgerblue", linestyles='dashed')
    #     self.scatter_n_gap += 1
        
    #     # scatter
    #     for sw in [False, True]:
    #         kwargs = {
    #             "attr_prefix"           : "fish_day", # 用來存取 self.[obj_prefix]_scatter_list 之類的變數
    #             "surf2mark_dict"        : day_mark_dict,
    #             "mapping_dict"          : fish_day_mark2str,
    #             "mark_style"            : 's',
    #             "separately"            : sw, # 分開畫
    #             "show_info"             : sw,
    #             "legend_name"           : "day",
    #             "legend_loc"            : 'upper right',
    #             "legend_bbox_to_anchor" : (1.0, 0.88) # 0 是靠左，1 是靠右
    #         }
    #         self.plot_marks_scatter(**kwargs)
            
    #         if not sw: self.scatter_n_gap += 1 # 畫完重疊的 scatter 後要加一次間距
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_cluster_boundary(self):
        
    #     # min_value line
    #     min_boundary = self.surface_area.squeeze().min()
    #     self.ax.axvline(x=min_boundary, color='k', linestyle='--')
    #     self.ax.text(min_boundary, 0.95, f'  x={min_boundary:.{self.digits}f}',
    #                      transform=self.ax.get_xaxis_transform(), ha='left',
    #                      color='black', path_effects=[self.text_path_effect])
        
    #     # cluster_max_value lines
    #     text_y_pos = {0: 0.92, 1: 0.95}
    #     for i, boundary in enumerate(self.cidx_max_area_dict.values()):
    #         self.ax.axvline(x=boundary, color='k', linestyle='--')
    #         self.ax.text(boundary, text_y_pos[(i%2)], f'x={boundary:.{self.digits}f}  ',
    #                      transform=self.ax.get_xaxis_transform(), ha='right',
    #                      color='black', path_effects=[self.text_path_effect])
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_cluster_count(self):
    #     key_list = list(self.cidx_max_area_dict.keys()) # [0, 2, 1]
    #     value_list = list(self.cidx_max_area_dict.values()) # [100, 200, 300]
    #     value_list.insert(0, self.surface_area.squeeze().min()) # [0, 100, 200, 300]
    #     for i, cluster in enumerate(key_list):
    #         text_center = (value_list[i+1] + value_list[i])/2
    #         # self.ax.axvline(x=text_center, color='green', linestyle='--')
    #         text = self.ax.text(text_center, 0.8, f'{self.cidx2clabel[cluster]}={self.clusters_count[cluster]}', 
    #                             transform=self.ax.get_xaxis_transform(), ha='center',
    #                             fontsize=16, color='#FFFFF2', path_effects=[self.text_path_effect])
    #         text.set_bbox(dict(boxstyle="round", pad=0.8, facecolor='#EE7785', alpha=0.7, edgecolor='none',
    #                            path_effects=[path_effects.withSimplePatchShadow(offset=(2, -2), foreground='black')]))
    #     # -------------------------------------------------------------------------------------
    
    
    # def plot_old_classdiv_boundary(self):
    #     for i, (key, value) in enumerate(self.old_classdiv_info_dict.items()):
    #         self.ax.axvline(x=value, color='r', linestyle='--', alpha=0.7)
    #         self.ax.text(value, 0.666, f'  {key:{self.digits}}:\n  {value:.{self.digits}f}', 
    #                      transform=self.ax.get_xaxis_transform(), ha='left',
    #                      color='red', path_effects=[self.text_path_effect], alpha=0.7)
    #     # -------------------------------------------------------------------------------------
    
    
    # def save_fig(self):
    #     self.fig.suptitle(self.fig_name, size=20)
    #     self.fig.savefig(str(self.clustered_xlsx_dir/ f"{{{self.clustered_xlsx_name}}}{'_kde' if self.x_axis_log_scale else ''}.png"))
    #     # -------------------------------------------------------------------------------------
    
    
    # def save_fig_with_old_classdiv(self):
    #     self.fig.suptitle(self.fig_name_old_classdiv, size=20)
    #     self.fig.savefig(str(self.compared_cluster_img_dir/ f"{self.fig_name_old_classdiv}.png"))
    #     # -------------------------------------------------------------------------------------
    
    
    
    # def plot_and_save_xlsx(self):
    #     # -------------------------------------------------------------------------------------
    #     self.run_kmeans()
    #     self.count_cluster_element()
    #     self.gen_surf2pred_dict()
    #     self.find_clusters_max_area()
    #     self.gen_clusters_idx2str()
    #     self.gen_clustered_xlsx_df()
    #     self.save_clustered_xlsx_df()
        
    #     # -------------------------------------------------------------------------------------
    #     self.plot_misc_settings()
    #     self.plot_hist()
    #     if self.x_axis_log_scale: self.plot_kde()
    #     # cluster
    #     self.plot_cluster_distribution()
    #     self.plot_cluster_boundary()
    #     self.plot_cluster_count()
    #     self.get_current_xlim()
    #     self.plot_batch_mark() # batch
    #     self.plot_day_mark() # day
    #     # legend
    #     self.ax.get_legend().remove()
    #     self.ax.add_artist(self.clusters_legend)
    #     self.ax.add_artist(self.fish_batch_legend)
    #     # self.ax.add_artist(self.fish_day_legend)
    #     # # colorbar
    #     # self.add_colorbar(self.y_kmeans, "cluster", self.clusters_cmap,
    #     #                   ticks=sorted(list(self.clusters_idx2str.keys())), 
    #     #                   ticklabels=list(self.labels)) # 如果 ticks 和 ticklabels 都有指定會自動將兩者 mapping
    #     # self.add_colorbar(self.fish_batch_mark, "batch", self.fish_batch_cmap, 
    #     #                   ticks=list(self.fish_batch_mark2str.keys()), 
    #     #                   ticklabels=list(self.fish_batch_mark2str.values()))
    #     self.add_colorbar(self.day_list, "day", self.fish_day_cmap, ticks=list(self.fish_day_cnt.keys()))
        
    #     self.save_fig()
        
    #     # -------------------------------------------------------------------------------------
    #     if self.old_classdiv_xlsx_path: self.plot_old_classdiv_boundary()
    #     if self.old_classdiv_xlsx_path: self.save_fig_with_old_classdiv()
    #     plt.close(self.fig)