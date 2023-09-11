import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from collections import OrderedDict
from copy import deepcopy
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from colorama import Fore, Back, Style

from .. import dname
from ..processeddatainstance import ProcessedDataInstance
from ...shared.clioutput import CLIOutput
from ...shared.config import load_config, dump_config
from ...shared.utils import create_new_dir
# -----------------------------------------------------------------------------/


class SurfaceAreaKMeansCluster():


    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self.processed_data_instance = ProcessedDataInstance()
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="SurfaceArea KMeans Cluster")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.processed_data_instance.set_attrs(config_file)
        self._set_config_attrs(config_file)
        
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
        # ---------------------------------------------------------------------/



    def _set_config_attrs(self, config_file:Union[str, Path]):
        """ Set below attributes
            - `self.batch_id_interval`: List[int]
            - `self.batch_idx2str`: Dict[int, str]
            - `self.random_seed`: int
            - `self.n_class`: int
            - `self.labels`: List[str]
            - `self.cluster_with_log_scale`: bool
            - `self.log_base`: int
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
        # ---------------------------------------------------------------------/



    def _set_orig_xlsx_attrs(self):
        """ Set below attributes
            - `self.orig_xlsx_path`: Path
            - `self.orig_xlsx_df`: pd.DataFrame
            - `self.orig_xlsx_path_split`: List[str]
        """
        def set_attr1() -> Path:
            """ `self.orig_xlsx_path` """
            if self.processed_data_instance.data_xlsx_path:
                return self.processed_data_instance.data_xlsx_path
            else:
                raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find `data.xlsx` please run `0.5.create_data_xlsx.py` to create it. {Style.RESET_ALL}\n")
        
        self.orig_xlsx_path: Path = set_attr1()
        self.orig_xlsx_df: pd.DataFrame = pd.read_excel(self.orig_xlsx_path, engine = 'openpyxl')
        self.orig_xlsx_path_split: List[str] = str(self.orig_xlsx_path).split(os.sep)
        # ---------------------------------------------------------------------/



    def _set_clustered_xlsx_attrs(self):
        """ Set below attributes
            - `self.clustered_xlsx_dir`: Path
            - `self.clustered_xlsx_name`: str
            - `self.clustered_xlsx_path`: Path
            - `self.clustered_xlsx_df` # assigned after `gen_clustered_xlsx_df()`
        """
        def set_attr1() -> str:
            """ `self.clustered_xlsx_name` """
            if self.cluster_with_log_scale:
                name = f"SURF{self.n_class}C_KMeansLOG{self.log_base}_RND{self.random_seed}"
            else:
                name = f"SURF{self.n_class}C_KMeansORIG_RND{self.random_seed}"
            return name
        
        def set_attr2() -> Path:
            """ `self.clustered_xlsx_dir` """
            if self.processed_data_instance.clustered_xlsx_dir:
                path = self.processed_data_instance.clustered_xlsx_dir.joinpath(self.clustered_xlsx_name)
            else:
                path = self.processed_data_instance.instance_root.joinpath("Clustered_xlsx", self.clustered_xlsx_name)
            create_new_dir(path)
            return path
            
        self.clustered_xlsx_name: str = set_attr1()
        self.clustered_xlsx_dir: Path = set_attr2()
        self.clustered_xlsx_path: Path = self.clustered_xlsx_dir.joinpath(f"{{{self.clustered_xlsx_name}}}_data.xlsx")
        self.clustered_xlsx_df = None # assigned after `gen_clustered_xlsx_df()`
        # ---------------------------------------------------------------------/



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
        # ---------------------------------------------------------------------/



    def _set_day_list(self):
        """
        """
        day_list: List[int] = deepcopy(self.bf_dnames)
        
        for i, bf_dname in enumerate(self.bf_dnames):
            fish_day = int(re.split(" |_|-", bf_dname)[3].replace("dpf", ""))
            day_list[i] = fish_day
        
        return day_list
        # ---------------------------------------------------------------------/



    def log(self, base, x):
        """
        """
        return np.log(x) / np.log(base)
        # ---------------------------------------------------------------------/



    def run_kmeans(self):
        """
        """
        if self.cluster_with_log_scale:
            self.surface_area = self.log(self.log_base, self.surface_area)
        
        self.kmeans.fit(self.surface_area)
        self.y_kmeans = self.kmeans.predict(self.surface_area) # 產生分群結果
        self.kmeans_centers = self.kmeans.cluster_centers_ # 取得群心
        
        if self.cluster_with_log_scale:
            self.surface_area   = self.log_base ** self.surface_area
            self.kmeans_centers = self.log_base ** self.kmeans_centers
        
        self._cli_out.write(f'kmeans_centers {type(self.kmeans_centers)}: \n{self.kmeans_centers}')
        # ---------------------------------------------------------------------/



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
        # ---------------------------------------------------------------------/



    def _set_cidx2clabel(self):
        """ execution dependency:
            - `self.cidx_max_area_dict`
        """
        self.cidx2clabel = {cidx: clabel for cidx, clabel in zip(self.cidx_max_area_dict.keys(), self.labels) }
        self._cli_out.write(f"self.cidx2clabel : {self.cidx2clabel}")
        # ---------------------------------------------------------------------/



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
        # ---------------------------------------------------------------------/



    def save_clustered_xlsx_df(self):
        """ execution dependency:
            - `self.clustered_xlsx_path`
            - `self.clustered_xlsx_df`
        """
        self.clustered_xlsx_df.to_excel(self.clustered_xlsx_path, engine="openpyxl", index=False)
        self._cli_out.write(f"Clustered XLSX : {self.clustered_xlsx_path.resolve()}")
        # ---------------------------------------------------------------------/



    def save_kmeans_centers(self):
        """
        """
        temp_dict = {i: center for i, center in enumerate(self.kmeans_centers.squeeze())}
        save_dict = {}
        
        for k, v in self.cidx2clabel.items():
            save_dict[v] = temp_dict[k]
        
        path = self.clustered_xlsx_dir.joinpath("kmeans_centers.toml")
        dump_config(path, save_dict)
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="0.6.cluster_data.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
        self.run_kmeans()
        self._set_cidx_max_area_dict()
        self._set_cidx2clabel()
        self.gen_clustered_xlsx_df()
        self.save_clustered_xlsx_df()
        self.save_kmeans_centers()
        # ---------------------------------------------------------------------/