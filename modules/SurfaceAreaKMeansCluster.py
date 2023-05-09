import os
import sys
import re
from typing import List
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict, Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans

from fileop import create_new_dir



class SurfaceAreaKMeansCluster():
    
    def __init__(self, xlsx_path:Path, n_clusters:int, label_str:List[str], kmeans_rnd:int, 
                 log_base:int=10, cluster_with_log_scale:bool=False, with_kde:bool=False,
                 old_classdiv_sheet_name:str=None) -> None:
        
        if isinstance(xlsx_path, Path): self.xlsx_path = xlsx_path
        else: raise TypeError("xlsx_path should be a 'Path' object, please using `from pathlib import Path`")
        self.xlsx_path_split_list = str(self.xlsx_path).split(os.sep)
        self.sinica_idx = None
        self.find_sinica_dir_in_path()
        self.dataset_id = self.xlsx_path_split_list[self.sinica_idx].split("_")[-1]
        self.xlsx_df = None
        self.surface_area = None # column: "Trunk surface area, SA (um2)"
        
        self.clustered_xlsx_dir = Path( os.sep.join(self.xlsx_path_split_list[:(self.sinica_idx+1)]) ) / r"{Modify}_xlsx"
        self.clustered_xlsx_name = (f"{n_clusters}CLS_SURF_"
                                    f"KMeans{f'LOG{log_base}' if cluster_with_log_scale else 'ORIG'}_"
                                    f"RND{kmeans_rnd}")
        self.clustered_xlsx_path = None
        self.clustered_xlsx_df = None
        
        self.old_classdiv_sheet_name = old_classdiv_sheet_name
        self.show_old_classdiv = True if self.old_classdiv_sheet_name is not None else False
        self.old_classdiv_info_dict = {}
        
        self.n_clusters = n_clusters
        self.cluster_with_log_scale = cluster_with_log_scale
        self.kmeans_rnd = kmeans_rnd
        self.kmeans = KMeans(n_clusters = self.n_clusters, random_state=self.kmeans_rnd)
        self.kmeans_centers = None
        self.y_kmeans = None # predict
        
        self.bins = None
        
        self.with_kde = with_kde
        self.log_base = log_base
        self.kde = None
        self.cluster_img_dir = Path( f"./k_means/cluster{'_with_kde' if self.with_kde else ''}/" ).joinpath(self.dataset_id)
        
        self.label_str = label_str # small -> big, e.g. ["S", "M", "L"]
        self.surf2pred_dict = None
        self.clusters_max_area = [0]*n_clusters
        self.clusters_count = None
        self.label_idx2str = None
        
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=200)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig_name = (f"{self.dataset_id}, {self.clustered_xlsx_name}"
                         f"{f', {self.old_classdiv_sheet_name}' if self.show_old_classdiv else ''}"
                         f"{', KDE' if self.with_kde else ''}")
        self.clusters_pt_y_pos = None # misc
        self.digits = None # misc
        self.text_path_effect = None
        self.plot_misc_settings()
    
    
    def __repr__(self):
        return (f'self.dataset_id              : {self.dataset_id}\n'
                f'self.xlsx_path               : {self.xlsx_path}\n'
                f'self.n_clusters              : {self.n_clusters}\n'
                f'self.label_str               : {self.label_str}\n'
                f'self.kmeans_rnd              : {self.kmeans_rnd}\n'
                f'self.log_base                : {self.log_base}\n'
                f'self.cluster_with_log_scale  : {self.cluster_with_log_scale}\n'
                f'self.with_kde                : {self.with_kde}\n'
                f'self.show_old_classdiv       : {self.show_old_classdiv}\n'
                f'self.old_classdiv_sheet_name : {self.old_classdiv_sheet_name}\n')
    
    
    def log(self, base, x):
        return np.log(x) / np.log(base)
    
    
    def find_sinica_dir_in_path(self): # To find `{ Reminder }_Academia_Sinica_i[num_RGB]` in `split_path`
        for i, text in enumerate(self.xlsx_path_split_list):
            if "Academia_Sinica" in text: self.sinica_idx = i
    
    
    def read_xlsx(self):
        if self.show_old_classdiv: self.xlsx_df :pd.DataFrame = pd.read_excel(self.xlsx_path, engine = 'openpyxl', sheet_name=self.old_classdiv_sheet_name)
        else: self.xlsx_df :pd.DataFrame = pd.read_excel(self.xlsx_path, engine = 'openpyxl') # read first sheet
        

    def extract_surface_area(self):
        self.surface_area = self.xlsx_df["Trunk surface area, SA (um2)"].to_numpy()
        self.surface_area = self.surface_area[:, None] # (100) -> (100, )

    
    def get_old_classdiv_info(self):
        if self.cluster_with_log_scale or self.with_kde:
            self.old_classdiv_info_dict['L_std_value'] = self.log(self.log_base, self.xlsx_df["L_1s"][0])
            self.old_classdiv_info_dict['avg_value']   = self.log(self.log_base, self.xlsx_df["average"][0])
            self.old_classdiv_info_dict['R_std_value'] = self.log(self.log_base, self.xlsx_df["R_1s"][0])
        else: 
            self.old_classdiv_info_dict['L_std_value'] = self.xlsx_df["L_1s"][0]
            self.old_classdiv_info_dict['avg_value']   = self.xlsx_df["average"][0]
            self.old_classdiv_info_dict['R_std_value'] = self.xlsx_df["R_1s"][0]
    
    
    def run_kmeans(self):
        if self.cluster_with_log_scale: self.surface_area = self.log(self.log_base, self.surface_area)
        self.kmeans.fit(self.surface_area)
        self.kmeans_centers = self.kmeans.cluster_centers_ # 群心
        print(f'kmeans_centers {type(self.kmeans_centers)}: \n{self.kmeans_centers}\n')
        self.y_kmeans = self.kmeans.predict(self.surface_area) # 分群
        if self.with_kde and (not self.cluster_with_log_scale):
            self.surface_area   = self.log(self.log_base, self.surface_area)
            self.kmeans_centers = self.log(self.log_base, self.kmeans_centers)
    
    
    def plot_hist(self):
        hist = self.ax.hist(self.surface_area, bins=100, density=True, alpha=0.7)
        density, self.bins, patches = hist
        widths = self.bins[1:] - self.bins[:-1]
        print(f"accum_p = {(density * widths).sum()}\n")

    
    def run_kde(self):
        # instantiate and fit the KDE model
        self.kde = KernelDensity(bandwidth=0.01178167723136119, kernel='gaussian')
        self.kde.fit(self.surface_area)

        # score_samples returns the log of the probability density
        logprob = self.kde.score_samples(self.bins[:, None])

        self.ax.fill_between(self.bins, np.exp(logprob), alpha=0.5, color="orange")
        # self.ax.plot(self.bins, np.exp(logprob), label='KDE', color="orange") # , linestyle='--'

    
    def count_cluster_element(self): # dependency: self.y_kmeans
        self.clusters_count = Counter(self.y_kmeans)
    
    
    def gen_surf2pred_dict(self): # dependency: self.y_kmeans
        self.surf2pred_dict = { area : label for area, label in zip(self.surface_area.squeeze(), self.y_kmeans)}
    
    
    def find_clusters_max_area(self): # dependency: self.surf2pred_dict
        for area, pred in self.surf2pred_dict.items():
            if area > self.clusters_max_area[pred]:
                self.clusters_max_area[pred] = area
        self.clusters_max_area = { cluster: max_area for cluster, max_area in enumerate(self.clusters_max_area)}
        self.clusters_max_area = OrderedDict(sorted(list(self.clusters_max_area.items()), key=lambda x: x[1]))
        print(f'self.clusters_max_area {type(self.clusters_max_area)}: {self.clusters_max_area}\n')
    
    
    def gen_label_idx2str(self): # dependency: self.clusters_max_area
        self.label_idx2str = { cls_idx: cls_str for (cls_idx, _), cls_str in zip(self.clusters_max_area.items(), self.label_str) }
        print(f'self.label_idx2str {type(self.label_idx2str)}: {self.label_idx2str}\n')
    
    
    def gen_clustered_xlsx_df(self): # dependency: self.label_idx2str
        col_class_dict = deepcopy(self.surf2pred_dict)
        for area, pred in col_class_dict.items():
            col_class_dict[area] = self.label_idx2str[pred]
        col_class_series = pd.Series(list(col_class_dict.values()), name="class")
        self.clustered_xlsx_df = pd.concat([self.xlsx_df, col_class_series], axis=1)
    
    
    def plot_misc_settings(self):
        self.fig.suptitle(self.fig_name, size=20)
        self.fig.subplots_adjust(top=0.9)
        
        if self.cluster_with_log_scale or self.with_kde: self.clusters_pt_y_pos = -0.5
        else: self.clusters_pt_y_pos = -2e-7
        
        if self.cluster_with_log_scale or self.with_kde: self.digits = 8
        else: self.digits = 2

        self.text_path_effect = path_effects.withSimplePatchShadow(offset=(0.5, -0.5), linewidth=1, foreground='black')
    
    
    def plot_cluster_distribution(self):
        self.ax.plot(self.surface_area, np.full_like(self.surface_area, self.clusters_pt_y_pos*0.4), '|k', 
                     markeredgewidth=1)
        self.ax.scatter(
            self.surface_area, np.full_like(self.surface_area, self.clusters_pt_y_pos),
            c = self.y_kmeans,    # 指定標記
            edgecolor = 'none',   # 無邊框
            # alpha = 0.5         # 不透明度
            cmap="viridis"
        )
    
    
    def plot_cluster_center(self):
        self.ax.scatter(self.kmeans_centers.reshape(-1), 
                        np.full_like(self.kmeans_centers.reshape(-1), self.clusters_pt_y_pos),
                        marker="x", s=50, color="black")
    
    
    def plot_cluster_boundary(self):
        for max_area in self.clusters_max_area.values():
            self.ax.axvline(x=max_area, color='k', linestyle='--')
            self.ax.text(max_area, 0.92, f'x={max_area:.{self.digits}f}  ',
                         transform=self.ax.get_xaxis_transform(), ha='right',
                         color='black', path_effects=[self.text_path_effect])
    
    
    def plot_cluster_count(self):
        key_list = list(self.clusters_max_area.keys()) # [0, 2, 1]
        value_list = list(self.clusters_max_area.values()) # [100, 200, 300]
        value_list.insert(0, self.surface_area.squeeze().min()) # [0, 100, 200, 300]
        for i, cluster in enumerate(key_list):
            text_center = (value_list[i+1] + value_list[i])/2
            # self.ax.axvline(x=text_center, color='green', linestyle='--')
            text = self.ax.text(text_center, 0.8, f'{self.label_idx2str[cluster]}={self.clusters_count[cluster]}', 
                                transform=self.ax.get_xaxis_transform(), ha='center',
                                fontsize=16, color='#FFFFF2', path_effects=[self.text_path_effect])
            text.set_bbox(dict(boxstyle="round", pad=0.8, facecolor='#EE7785', alpha=0.7, edgecolor='none',
                               path_effects=[path_effects.withSimplePatchShadow(offset=(2, -2), foreground='black')]))
    
    
    def plot_old_classdiv_boundary(self):
        for i, (key, value) in enumerate(self.old_classdiv_info_dict.items()):
            self.ax.axvline(x=value, color='r', linestyle='--', alpha=0.7)
            self.ax.text(value, 0.22*(i+1), f'  {key}:\n  {value:.{self.digits}f}', 
                         transform=self.ax.get_xaxis_transform(), ha='left',
                         color='red', path_effects=[self.text_path_effect], alpha=0.7)
    
    
    def save_fig(self):
        create_new_dir(str(self.clustered_xlsx_dir), display_in_CLI=False)
        self.fig.savefig(str(self.clustered_xlsx_dir/ f"{{{self.clustered_xlsx_name}}}{'_kde' if self.with_kde else ''}.png"))
    
    
    def save_fig_with_old_classdiv(self):
        create_new_dir(str(self.cluster_img_dir), display_in_CLI=False)
        self.fig.savefig(str(self.cluster_img_dir/ f"{self.fig_name}.png"))
        plt.close(self.fig)
    
    
    def save_clustered_xlsx_df(self):
        create_new_dir(str(self.clustered_xlsx_dir), display_in_CLI=False)
        self.clustered_xlsx_path = self.clustered_xlsx_dir / f"{{{self.clustered_xlsx_name}}}_data.xlsx"
        if not os.path.exists(str(self.clustered_xlsx_path)):
            self.clustered_xlsx_df.to_excel(str(self.clustered_xlsx_path), engine="openpyxl", index=False)
    
    
    def plot_and_save_xlsx(self):
        self.read_xlsx()
        self.extract_surface_area()
        if self.show_old_classdiv: self.get_old_classdiv_info()
        self.run_kmeans()
        self.plot_hist()
        if self.with_kde: self.run_kde()
        self.count_cluster_element()
        self.gen_surf2pred_dict()
        self.find_clusters_max_area()
        self.gen_label_idx2str()
        self.gen_clustered_xlsx_df()
        self.plot_cluster_distribution()
        self.plot_cluster_center()
        self.plot_cluster_boundary()
        self.plot_cluster_count()
        self.save_fig()
        if self.show_old_classdiv: self.plot_old_classdiv_boundary()
        if self.show_old_classdiv: self.save_fig_with_old_classdiv()
        self.save_clustered_xlsx_df()