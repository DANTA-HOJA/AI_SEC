import os
import sys
import re
from typing import List, Dict, Union
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict, Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans

from fileop import create_new_dir



class SurfaceAreaKMeansCluster():
    
    def __init__(self, xlsx_path:Path, n_clusters:int, clusters_str:List[str], kmeans_rnd:int, 
                 log_base:int=10, cluster_with_log_scale:bool=False, x_axis_log_scale:bool=False,
                 old_classdiv_xlsx_path:Path=None) -> None:
        # -------------------------------------------------------------------------------------
        if isinstance(xlsx_path, Path): 
            self.orig_xlsx_path = xlsx_path
        else: raise TypeError("`xlsx_path` should be a `Path` object, please using `from pathlib import Path`")
        
        self.orig_xlsx_path_split = str(self.orig_xlsx_path).split(os.sep)
        self.orig_xlsx_df: pd.DataFrame = pd.read_excel(self.orig_xlsx_path, engine = 'openpyxl')
        self.surface_area = self.orig_xlsx_df["Trunk surface area, SA (um2)"].to_numpy()[:, None] # reshape: (100) -> (100, )
        self.sinica_dir_idx = None
        self.find_sinica_dir_in_path()
        self.dataset_id = self.orig_xlsx_path_split[self.sinica_dir_idx].split("_")[-1] # e.g. i409, i505
        
        self.fish_dname = list(self.orig_xlsx_df["Posterior (SP8, .tif)"])
        self.fish_batch_divnum = [0, 116, 164, 207, 255] # n1 < x <= n2
        self.fish_batch_mark2str = {0: "i162", 1:"i242", 2:"i409", 3: "i505"}
        self.fish_batch_mark = None # 建立一個 list 標記 data 對應的 batch
        self.get_fish_batch_info()
        self.fish_day_mark = None # 建立一個 list 標記 data 對應的 day (dpf)
        self.fish_day_cnt = None
        self.get_fish_day_info()
        
        # -------------------------------------------------------------------------------------
        self.clustered_xlsx_dir = Path( os.sep.join(self.orig_xlsx_path_split[:(self.sinica_dir_idx+1)]) ) / r"{Modify}_xlsx"
        self.clustered_xlsx_name = (f"{n_clusters}CLS_SURF_"
                                    f"KMeans{f'LOG{log_base}' if cluster_with_log_scale else 'ORIG'}_"
                                    f"RND{kmeans_rnd}")
        self.clustered_xlsx_path = None
        self.clustered_xlsx_df = None
        
        # -------------------------------------------------------------------------------------
        self.show_old_classdiv = None
        self.old_classdiv_xlsx_path = None
        self.old_classdiv_xlsx_df = None
        self.old_classdiv_strategy = None
        self.old_classdiv_info_dict = {}
        
        if old_classdiv_xlsx_path is not None:
            if isinstance(old_classdiv_xlsx_path, Path):
                self.old_classdiv_xlsx_path = old_classdiv_xlsx_path
                self.show_old_classdiv = True
            else: raise TypeError("`old_classdiv_xlsx_path` should be a `Path` object, please using `from pathlib import Path`")
        
        # -------------------------------------------------------------------------------------
        self.cluster_with_log_scale = cluster_with_log_scale
        self.log_base = log_base
        self.x_axis_log_scale = x_axis_log_scale
        
        # -------------------------------------------------------------------------------------
        self.n_clusters = n_clusters
        self.kmeans_rnd = kmeans_rnd
        self.kmeans = KMeans(n_clusters = self.n_clusters, random_state=self.kmeans_rnd)
        self.kmeans_centers = None
        self.y_kmeans = None # predict
        
        # -------------------------------------------------------------------------------------
        self.bins = None # x position for kde
        self.kde = None
        self.kde_kwargs = {"bandwidth": 0.01178167723136119, "kernel": 'gaussian'}
        self.compared_cluster_img_dir = self.clustered_xlsx_dir.joinpath("compare_clustering", "KMeans_comp_STDEV",
                                                                         f"{f'x-axis in LOG{self.log_base} scale' if self.x_axis_log_scale else 'x-axis in ORIG scale'}" )
        
        # -------------------------------------------------------------------------------------
        self.clusters_str = clusters_str # small -> big, e.g. ["S", "M", "L"]
        self.surf2pred_dict = None
        self.clusters_max_area = [0]*n_clusters
        self.clusters_count = None
        self.clusters_idx2str = None
        
        # -------------------------------------------------------------------------------------
        if self.show_old_classdiv: self.get_old_classdiv_info()
        
        self.fig = plt.figure(figsize=(16, 16), dpi=200) # figure 內的 element 都是用比例決定位置的
                                                         #  一旦決定 figsize 之後如果要調整圖片"長寬比"最好只調整一邊
                                                         #  調整"整張圖片大小" -> dpi
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.divider = make_axes_locatable(self.ax)
        self.fig_name = f"{self.dataset_id}, {self.clustered_xlsx_name}{', KDE' if self.x_axis_log_scale else ''}"
        self.fig_name_old_classdiv = f"{self.fig_name}, {self.old_classdiv_strategy}_STDEV"
        self.fig.suptitle(self.fig_name, size=20)
        self.fig.subplots_adjust(top=0.9)
        
        self.ax_x_lim = None # 記憶 self.ax.get_xlim()
        self.scatter_init_pos = None # misc
        self.scatter_n_gap = 0 # 各 scatter 的間距倍數: self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)
        self.digits = None # misc
        self.text_path_effect = None # misc
        
        self.colorbar_n_gap = 0
        self.clusters_cmap = "viridis"
        self.fish_batch_cmap = "Paired"
        self.fish_day_cmap = "Dark2"
        self.clusters_legend = None
        self.fish_batch_legend = None
        self.fish_day_legend = None
        # -------------------------------------------------------------------------------------
    
    
    def __repr__(self):
        part1 = (f'self.dataset_id              : {self.dataset_id}\n'
                 f'self.orig_xlsx_path          : {self.orig_xlsx_path}\n'
                 f'self.n_clusters              : {self.n_clusters}\n'
                 f'self.clusters_str            : {self.clusters_str}\n'
                 f'self.kmeans_rnd              : {self.kmeans_rnd}\n'
                 f'self.log_base                : {self.log_base}\n'
                 f'self.cluster_with_log_scale  : {self.cluster_with_log_scale}\n'
                 f'self.x_axis_log_scale        : {self.x_axis_log_scale}\n')
        part2 = (f'self.old_classdiv_xlsx_path  : {self.old_classdiv_xlsx_path}\n'
                 f'self.old_classdiv_strategy   : {self.old_classdiv_strategy}\n')
        output = (part1 + part2) if self.show_old_classdiv else part1
        
        return output
        # -------------------------------------------------------------------------------------
    
    
    def log(self, base, x):
        return np.log(x) / np.log(base)
        # -------------------------------------------------------------------------------------
    
    
    def find_sinica_dir_in_path(self): # To find `{ Reminder }_Academia_Sinica_i[num_RGB]` in `split_path`
        for i, text in enumerate(self.orig_xlsx_path_split):
            if "Academia_Sinica" in text: self.sinica_dir_idx = i
        # -------------------------------------------------------------------------------------

    
    def get_fish_batch_info(self):
        self.fish_batch_mark = deepcopy(self.fish_dname)
        for i, fish_dname in enumerate(self.fish_dname):
            fish_dname_id = int(re.split(" |_|-", fish_dname)[8])
            for j in range(len(self.fish_batch_divnum)-1):
                if (fish_dname_id > self.fish_batch_divnum[j]) and (fish_dname_id <= self.fish_batch_divnum[j+1]):
                    self.fish_batch_mark[i] = list(self.fish_batch_mark2str.keys())[j]
                    break
        # -------------------------------------------------------------------------------------
    
    
    def get_fish_day_info(self):
        self.fish_day_mark = deepcopy(self.fish_dname)
        for i, fish_dname in enumerate(self.fish_dname):
            fish_day = int(re.split(" |_|-", fish_dname)[3].replace("dpf", ""))
            self.fish_day_mark[i] = fish_day
        # -------------------------------------------------------------------------------------
    
    
    def get_old_classdiv_info(self):
        self.old_classdiv_xlsx_df: pd.DataFrame = pd.read_excel(self.old_classdiv_xlsx_path, engine = 'openpyxl')
        old_classdiv_xlsx_name = str(self.old_classdiv_xlsx_path).split(os.sep)[-1] # '{3CLS_SURF_050STDEV}_data.xlsx'
        old_classdiv_xlsx_name_split = re.split("{|_|}", old_classdiv_xlsx_name) # ['', '3CLS', 'SURF', '050STDEV', '', 'data.xlsx']
        old_classdiv_num = int(old_classdiv_xlsx_name_split[1].replace("CLS", ""))
        assert old_classdiv_num == self.n_clusters, (f"n clusters in `old_classdiv_xlsx` are not match, "
                                                     f"expect {self.n_clusters}, but got {old_classdiv_num}")
        self.old_classdiv_strategy = int(old_classdiv_xlsx_name_split[3].replace("STDEV", ""))/100 # '050STDEV' -> 0.5
        
        if self.x_axis_log_scale:
            self.old_classdiv_info_dict['L_std_value'] = self.log(self.log_base, self.old_classdiv_xlsx_df["L_1s"][0])
            self.old_classdiv_info_dict['avg_value']   = self.log(self.log_base, self.old_classdiv_xlsx_df["average"][0])
            self.old_classdiv_info_dict['R_std_value'] = self.log(self.log_base, self.old_classdiv_xlsx_df["R_1s"][0])
        else: 
            self.old_classdiv_info_dict['L_std_value'] = self.old_classdiv_xlsx_df["L_1s"][0]
            self.old_classdiv_info_dict['avg_value']   = self.old_classdiv_xlsx_df["average"][0]
            self.old_classdiv_info_dict['R_std_value'] = self.old_classdiv_xlsx_df["R_1s"][0]
        # -------------------------------------------------------------------------------------
    
    
    def run_kmeans(self):
        if self.cluster_with_log_scale: self.surface_area = self.log(self.log_base, self.surface_area)
        self.kmeans.fit(self.surface_area)
        self.kmeans_centers = self.kmeans.cluster_centers_ # 群心
        print(f'kmeans_centers {type(self.kmeans_centers)}: \n{self.kmeans_centers}\n')
        self.y_kmeans = self.kmeans.predict(self.surface_area) # 產生分群結果
        
        if (not self.x_axis_log_scale) and self.cluster_with_log_scale: # 還原回 x-axis 的原始刻度（量級）
            self.surface_area   = self.log_base ** self.surface_area
            self.kmeans_centers = self.log_base ** self.kmeans_centers
        
        if self.x_axis_log_scale and (not self.cluster_with_log_scale): # 將 x-axis 的原始刻度（量級）取 LOG
            self.surface_area   = self.log(self.log_base, self.surface_area)
            self.kmeans_centers = self.log(self.log_base, self.kmeans_centers)
        # -------------------------------------------------------------------------------------

    
    def count_cluster_element(self): # dependency: self.y_kmeans
        self.clusters_count = Counter(self.y_kmeans)
        # -------------------------------------------------------------------------------------
    
    
    def gen_surf2pred_dict(self): # dependency: self.y_kmeans
        self.surf2pred_dict = { area : label for area, label in zip(self.surface_area.squeeze(), self.y_kmeans)}
        # -------------------------------------------------------------------------------------
    
    
    def find_clusters_max_area(self): # dependency: self.surf2pred_dict
        for area, pred in self.surf2pred_dict.items():
            if area > self.clusters_max_area[pred]:
                self.clusters_max_area[pred] = area
        self.clusters_max_area = { cluster: max_area for cluster, max_area in enumerate(self.clusters_max_area)}
        self.clusters_max_area = OrderedDict(sorted(list(self.clusters_max_area.items()), key=lambda x: x[1]))
        print(f'self.clusters_max_area {type(self.clusters_max_area)}: {self.clusters_max_area}\n')
        # -------------------------------------------------------------------------------------
    
    
    def gen_clusters_idx2str(self): # dependency: self.clusters_max_area
        self.clusters_idx2str = { cls_idx: cls_str for (cls_idx, _), cls_str in zip(self.clusters_max_area.items(), self.clusters_str) }
        print(f'self.clusters_idx2str {type(self.clusters_idx2str)}: {self.clusters_idx2str}\n')
        # -------------------------------------------------------------------------------------
    
    
    def gen_clustered_xlsx_df(self): # dependency: self.clusters_idx2str
        col_class_dict = deepcopy(self.surf2pred_dict)
        for area, pred in col_class_dict.items():
            col_class_dict[area] = self.clusters_idx2str[pred]
        # class
        new_series = pd.Series(list(col_class_dict.values()), name="class")
        self.clustered_xlsx_df = pd.concat([self.orig_xlsx_df, new_series], axis=1)
        # batch
        new_list = [ self.fish_batch_mark2str[batch_mark] for batch_mark in self.fish_batch_mark ]
        new_series = pd.Series(new_list, name="batch")
        self.clustered_xlsx_df = pd.concat([self.clustered_xlsx_df, new_series], axis=1)
        # day
        new_series = pd.Series(self.fish_day_mark, name="day")
        self.clustered_xlsx_df = pd.concat([self.clustered_xlsx_df, new_series], axis=1)
        # -------------------------------------------------------------------------------------
    
    
    def plot_misc_settings(self):
        
        if self.x_axis_log_scale: self.scatter_init_pos = -0.175
        else: self.scatter_init_pos = -8e-8
        
        if self.x_axis_log_scale: self.digits = 8
        else: self.digits = 2

        self.text_path_effect = path_effects.withSimplePatchShadow(
                                    offset=(0.5, -0.5), linewidth=1, foreground='black')
        # -------------------------------------------------------------------------------------
    
    
    def get_current_xlim(self):
        self.ax_x_lim = self.ax.get_xlim()
        # -------------------------------------------------------------------------------------
    
    
    def add_colorbar(self, mapping_list:list, name:str, cmap:str,
                           ticks:List, ticklabels:Union[List[str], None]=None):
        
        cax = self.divider.append_axes("right", "2%", pad=0.3+self.colorbar_n_gap*0.2) # "右側" 加新的軸
        self.colorbar_n_gap += 1                                                       # init: self.divider = make_axes_locatable(self.ax)
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array(mapping_list)  # 會自動統計 items 並排序
        cbar = self.fig.colorbar(mappable, cax=cax) # create a `color_bar`
        cbar.ax.set_xlabel(name, labelpad=10)  # 設置 `color_bar` 的標籤
        cbar.set_ticks(ticks, labels=ticklabels)
        # -------------------------------------------------------------------------------------
    
    
    def plot_hist(self):
        hist = self.ax.hist(self.surface_area, bins=100, density=True, alpha=0.7)
        density, self.bins, patches = hist
        widths = self.bins[1:] - self.bins[:-1]
        print(f"accum_p = {(density * widths).sum()}\n")
        # -------------------------------------------------------------------------------------
    
    
    def plot_kde(self):
        # instantiate and fit the KDE model
        self.kde = KernelDensity(**self.kde_kwargs)
        self.kde.fit(self.surface_area)

        # score_samples returns the log of the probability density
        logprob = self.kde.score_samples(self.bins[:, None])

        self.ax.fill_between(self.bins, np.exp(logprob), alpha=0.5, color="orange")
        # self.ax.plot(self.bins, np.exp(logprob), label='KDE', color="orange") # , linestyle='--'
        # -------------------------------------------------------------------------------------
    
    
    def plot_cluster_distribution(self):
        # black_ticks
        self.ax.plot(self.surface_area, np.full_like(self.surface_area, self.scatter_init_pos), 
                     '|k', markeredgewidth=1)
        # cluster
        scatter = self.ax.scatter(
            self.surface_area, np.full_like(self.surface_area, self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
            c = self.y_kmeans,    # 指定標記
            edgecolor = 'none',   # 無邊框
            # alpha = 0.5         # 不透明度
            cmap=self.clusters_cmap,
        )
        # cluster_center
        self.ax.scatter(self.kmeans_centers.reshape(-1), 
                        np.full_like(self.kmeans_centers.reshape(-1), self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
                        marker="x", s=50, color="black")
        self.scatter_n_gap += 1
        # legend
        legend_labels = OrderedDict(sorted(list(self.clusters_idx2str.items()), key=lambda x: x[0])).values() # sort by cluster_idx
        self.clusters_legend = self.ax.legend(handles=scatter.legend_elements()[0],
                                             labels=list(legend_labels),
                                             title='cluster', loc='upper right', bbox_to_anchor=(1, 0.99))
        # -------------------------------------------------------------------------------------
    
    
    def plot_batch_mark(self):
        
        batch_mark_dict = { area: batch_mark for area, batch_mark in zip(self.surface_area.squeeze(), self.fish_batch_mark) }
        cmap = cm.get_cmap(self.fish_batch_cmap)
        colors = cmap(np.linspace(0, 1, len(self.fish_batch_mark2str)))
        
        scatter_list = []
        for i, batch in enumerate(self.fish_batch_mark2str.keys()):
            
            surface_area_list = []
            batch_mark_list = []
            for area, batch_mark in batch_mark_dict.items():
                if batch_mark == batch:
                    surface_area_list.append(area)
                    batch_mark_list.append(batch_mark)
        
            scatter = self.ax.scatter(
                surface_area_list, np.full_like(surface_area_list, self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
                edgecolor = 'none',   # 無邊框
                # alpha = 0.5         # 不透明度
                color=colors[i],
                label=self.fish_batch_mark2str[i],
            ); self.scatter_n_gap += 1
            
            scatter_list.append(scatter)
            print(f"batch = {batch}, {len(batch_mark_list)}")
        
        # collect and create the legend set of scatters
        self.fish_batch_legend = self.ax.legend(handles=scatter_list, title='batch', loc='upper left', bbox_to_anchor=(0, 0.99))
        # -------------------------------------------------------------------------------------
    
    
    def plot_day_mark(self):
        
        self.fish_day_cnt = Counter(self.fish_day_mark)
        self.fish_day_cnt = OrderedDict(sorted(self.fish_day_cnt.items(), key=lambda x: x[0])) # sort by day
        day_mark_dict = { area: day_mark for area, day_mark in zip(self.surface_area.squeeze(), self.fish_day_mark) }
        cmap = cm.get_cmap(self.fish_day_cmap)
        colors = cmap(np.linspace(0, 1, len(self.fish_day_cnt)))
        
        scatter_list = []
        for i, day in enumerate(self.fish_day_cnt.keys()):
            
            surface_area_list = []
            day_mark_list = []
            for area, day_mark in day_mark_dict.items():
                if day_mark == day:
                    surface_area_list.append(area)
                    day_mark_list.append(day_mark)
            
            scatter = self.ax.scatter(
                surface_area_list, np.full_like(surface_area_list, self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
                edgecolor = 'none',   # 無邊框
                # alpha = 0.5         # 不透明度
                color=colors[i],
                label=f"{day}",
                marker='s'
            ); self.scatter_n_gap += 1
            
            scatter_list.append(scatter)
            print(f"day = {day}, {len(day_mark_list)}")
        
        # collect and create the legend set of scatters
        self.fish_day_legend = self.ax.legend(handles=scatter_list, title='day', loc='upper right', bbox_to_anchor=(1, 0.98))
        # -------------------------------------------------------------------------------------
    
    
    def plot_marks_overlay(self, mark_list:List, mark_style:str, cmap:str):
        
        # dividing line
        self.ax.hlines(self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83), 
                       self.ax_x_lim[0], self.ax_x_lim[1], 
                       color="dodgerblue", linestyles='dashed')
        self.scatter_n_gap += 1
        # scatter
        self.ax.scatter(
            self.surface_area, np.full_like(self.surface_area, self.scatter_init_pos*(2.85+self.scatter_n_gap*1.83)),
            c = mark_list,    # 指定標記
            edgecolor = 'none',   # 無邊框
            # alpha = 0.5         # 不透明度
            cmap=cmap,
            marker=mark_style
        ); self.scatter_n_gap += 1
        # -------------------------------------------------------------------------------------
    
    
    def plot_cluster_boundary(self):
        
        # min_value line
        min_value = self.surface_area.squeeze().min()
        self.ax.axvline(x=min_value, color='k', linestyle='--')
        self.ax.text(min_value, 0.92, f'  x={min_value:.{self.digits}f}',
                         transform=self.ax.get_xaxis_transform(), ha='left',
                         color='black', path_effects=[self.text_path_effect])
        
        # cluster_max_value lines
        for max_area in self.clusters_max_area.values():
            self.ax.axvline(x=max_area, color='k', linestyle='--')
            self.ax.text(max_area, 0.92, f'x={max_area:.{self.digits}f}  ',
                         transform=self.ax.get_xaxis_transform(), ha='right',
                         color='black', path_effects=[self.text_path_effect])
        # -------------------------------------------------------------------------------------
    
    
    def plot_cluster_count(self):
        key_list = list(self.clusters_max_area.keys()) # [0, 2, 1]
        value_list = list(self.clusters_max_area.values()) # [100, 200, 300]
        value_list.insert(0, self.surface_area.squeeze().min()) # [0, 100, 200, 300]
        for i, cluster in enumerate(key_list):
            text_center = (value_list[i+1] + value_list[i])/2
            # self.ax.axvline(x=text_center, color='green', linestyle='--')
            text = self.ax.text(text_center, 0.8, f'{self.clusters_idx2str[cluster]}={self.clusters_count[cluster]}', 
                                transform=self.ax.get_xaxis_transform(), ha='center',
                                fontsize=16, color='#FFFFF2', path_effects=[self.text_path_effect])
            text.set_bbox(dict(boxstyle="round", pad=0.8, facecolor='#EE7785', alpha=0.7, edgecolor='none',
                               path_effects=[path_effects.withSimplePatchShadow(offset=(2, -2), foreground='black')]))
        # -------------------------------------------------------------------------------------
    
    
    def plot_old_classdiv_boundary(self):
        for i, (key, value) in enumerate(self.old_classdiv_info_dict.items()):
            self.ax.axvline(x=value, color='r', linestyle='--', alpha=0.7)
            self.ax.text(value, 0.666, f'  {key:{self.digits}}:\n  {value:.{self.digits}f}', 
                         transform=self.ax.get_xaxis_transform(), ha='left',
                         color='red', path_effects=[self.text_path_effect], alpha=0.7)
        # -------------------------------------------------------------------------------------
    
    
    def save_fig(self):
        create_new_dir(str(self.clustered_xlsx_dir), display_in_CLI=False)
        self.fig.savefig(str(self.clustered_xlsx_dir/ f"{{{self.clustered_xlsx_name}}}{'_kde' if self.x_axis_log_scale else ''}.png"))
        # -------------------------------------------------------------------------------------
    
    
    def save_fig_with_old_classdiv(self):
        create_new_dir(str(self.compared_cluster_img_dir), display_in_CLI=False)
        self.fig.suptitle(self.fig_name_old_classdiv, size=20)
        self.fig.savefig(str(self.compared_cluster_img_dir/ f"{self.fig_name_old_classdiv}.png"))
        # -------------------------------------------------------------------------------------
    
    
    def save_clustered_xlsx_df(self):
        create_new_dir(str(self.clustered_xlsx_dir), display_in_CLI=False)
        self.clustered_xlsx_path = self.clustered_xlsx_dir / f"{{{self.clustered_xlsx_name}}}_data.xlsx"
        self.clustered_xlsx_df.to_excel(str(self.clustered_xlsx_path), engine="openpyxl", index=False)
        # -------------------------------------------------------------------------------------
    
    
    def plot_and_save_xlsx(self):
        # -------------------------------------------------------------------------------------
        self.run_kmeans()
        self.count_cluster_element()
        self.gen_surf2pred_dict()
        self.find_clusters_max_area()
        self.gen_clusters_idx2str()
        self.gen_clustered_xlsx_df()
        self.save_clustered_xlsx_df()
        
        # -------------------------------------------------------------------------------------
        self.plot_misc_settings()
        self.plot_hist()
        if self.x_axis_log_scale: self.plot_kde()
        # cluster
        self.plot_cluster_distribution()
        self.plot_cluster_boundary()
        self.plot_cluster_count()
        self.get_current_xlim()
        # batch
        self.plot_marks_overlay(self.fish_batch_mark, 'o', self.fish_batch_cmap)
        self.plot_batch_mark()
        # day
        self.plot_marks_overlay(self.fish_day_mark, 's', self.fish_day_cmap)
        self.plot_day_mark()
        # legend
        self.ax.get_legend().remove()
        self.ax.add_artist(self.clusters_legend)
        self.ax.add_artist(self.fish_batch_legend)
        # self.ax.add_artist(self.fish_day_legend)
        # # colorbar
        # self.add_colorbar(self.y_kmeans, "cluster", self.clusters_cmap,
        #                   ticks=list(self.clusters_idx2str.keys()), 
        #                   ticklabels=list(self.clusters_idx2str.values()))
        # self.add_colorbar(self.fish_batch_mark, "batch", self.fish_batch_cmap, 
        #                   ticks=list(self.fish_batch_mark2str.keys()), 
        #                   ticklabels=list(self.fish_batch_mark2str.values()))
        self.add_colorbar(self.fish_day_mark, "day", self.fish_day_cmap, ticks=list(self.fish_day_cnt.keys()))
        
        self.save_fig()
        
        # -------------------------------------------------------------------------------------
        if self.show_old_classdiv: self.plot_old_classdiv_boundary()
        if self.show_old_classdiv: self.save_fig_with_old_classdiv()
        plt.close(self.fig)