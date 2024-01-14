import os
import re
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from colorama import Back, Fore, Style
from sklearn.cluster import KMeans

from ...shared.baseobject import BaseObject
from ...shared.config import dump_config
from ...shared.utils import create_new_dir
from .. import dname
from ..processeddatainstance import ProcessedDataInstance
from .utils import log
# -----------------------------------------------------------------------------/


class SurfaceAreaKMeansCluster(BaseObject):

    def __init__(self, processed_data_instance:ProcessedDataInstance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("SurfaceArea KMeans Cluster")
        
        if processed_data_instance:
            self._processed_di = processed_data_instance
        else:
            self._processed_di = ProcessedDataInstance()
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super()._set_attrs(config)
        self._processed_di.parse_config(config)
        
        self._set_orig_df()
        self._set_clustered_file_attrs()
        
        self.bf_dnames:List[str] = list(self.orig_df["Brightfield"])
        self.surface_area = self.orig_df["Trunk surface area, SA (um2)"].to_numpy()[:, None] # reshape: (100) -> (100, )
        
        # train_df ( for KMeans training )
        self.train_df = self.orig_df[(self.orig_df["dataset"] == "train")]
        self.train_sa = self.train_df["Trunk surface area, SA (um2)"].to_numpy()[:, None] # reshape: (100) -> (100, )
        
        # others informations
        self.batch_idx_list:List[int] = self._set_batch_idx_list() # data 對應的 batch index
        self.day_list:List[int] = self._set_day_list() # data 對應的 day (dpf)
        
        self.kmeans = KMeans(n_clusters = self.n_class, random_state=self.random_seed)
        self.kmeans_centers = None # assigned after `run_kmeans()`
        self.sa_y = None # assigned after `run_kmeans()`
        self.cidx_max_area_dict = None # assigned after `_set_cidx_max_area_dict()`
        self.cidx2clabel = None # assigned after `_set_cidx2clabel()`
        self.clustered_df = None # assigned after `_gen_clustered_df()`
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """ Set below attributes
            - `self.batch_id_interval`: List[int]
            - `self.batch_idx2str`: Dict[int, str]
            - `self.random_seed`: int
            - `self.n_class`: int
            - `self.labels`: List[str]
            - `self.cluster_with_log_scale`: bool
            - `self.log_base`: int
        """        
        """ [batch_info] """
        self.batch_id_interval: List[int] = self.config["batch_info"]["id_interval"]
        self.batch_idx2str: Dict[int, str] = \
            {i: name for i, name in enumerate(self.config["batch_info"]["names"])}
        
        """ [cluster] """
        self.random_seed: int = self.config["cluster"]["random_seed"]
        self.n_class: int = self.config["cluster"]["n_class"]
        self.labels: List[str] = self.config["cluster"]["labels"]
        self.cluster_with_log_scale: bool = self.config["cluster"]["cluster_with_log_scale"]
        
        """ [log_scale] """
        self.log_base: int = self.config["log_scale"]["base"]
        # ---------------------------------------------------------------------/


    def _set_orig_df(self):
        """
        """
        basename = f"datasplit_{self.random_seed}.csv"
        path = self._processed_di.instance_root.joinpath(basename)
        
        if path.exists():
            self._cli_out.write(f"{basename} : '{path}'")
        else:
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find `{basename}`, "
                                    f"please run `0.5.2.split_data.py` to create it. {Style.RESET_ALL}\n")
        
        self.orig_df = pd.read_csv(path, encoding='utf_8_sig', index_col=[0])
        # ---------------------------------------------------------------------/


    def _set_clustered_file_attrs(self):
        """ Set below attributes
            - self.clustered_file
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
        self.dst_root = self._processed_di.instance_root.joinpath("Clustered_File", desc)
        create_new_dir(self.dst_root)
        
        self.clustered_file: Path = self.dst_root.joinpath(f"{{{desc}}}_datasplit.csv")
        # ---------------------------------------------------------------------/


    def _set_batch_idx_list(self) -> List[int]:
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


    def _set_day_list(self) -> List[int]:
        """
        """
        day_list: List[int] = deepcopy(self.bf_dnames)
        
        for i, bf_dname in enumerate(self.bf_dnames):
            fish_day = int(re.split(" |_|-", bf_dname)[3].replace("dpf", ""))
            day_list[i] = fish_day
        
        return day_list
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        self._run_kmeans()
        self._set_cidx_max_area_dict()
        self._set_cidx2clabel()
        self._gen_clustered_df()
        self._save_clustered_df()
        self._save_kmeans_centers()
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _run_kmeans(self):
        """
        """
        if self.cluster_with_log_scale:
            self.surface_area = log(self.log_base, self.surface_area)
            self.train_sa = log(self.log_base, self.train_sa)
        
        self.kmeans.fit(self.train_sa) # 僅利用 train set 訓練 KMeans
        self.kmeans_centers = self.kmeans.cluster_centers_ # 取得群心
        self.sa_y = self.kmeans.predict(self.surface_area) # 產生所有分群結果
        
        if self.cluster_with_log_scale:
            self.surface_area   = self.log_base ** self.surface_area
            self.train_sa = self.log_base ** self.train_sa
            self.kmeans_centers = self.log_base ** self.kmeans_centers
        
        self._cli_out.write(f'kmeans_centers {type(self.kmeans_centers)}: \n{self.kmeans_centers}')
        # ---------------------------------------------------------------------/


    def _set_cidx_max_area_dict(self):
        """ execution dependency:
            - `self.n_class`
            - `self.surface_area`
            - `self.sa_y`
        """
        cidx_max_area_list = [0]*self.n_class
        for area, cidx in zip(self.surface_area.squeeze(), self.sa_y):
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


    def _gen_clustered_df(self):
        """ execution dependency:
            - `self.orig_df`
            - `self.sa_y`
            - `self.cidx2clabel`
            - `self.batch_idx_list`
            - `self.batch_idx2str`
            - `self.day_list`
        """
        self.clustered_df = self.orig_df.copy()
        
        """ Add class column to `self.orig_df` """
        for idx, cidx in zip(self.orig_df.index, self.sa_y):
            self.clustered_df.loc[idx, "class"] = self.cidx2clabel[cidx]
        
        # reset index
        self.clustered_df = self.clustered_df.reset_index()
        self.clustered_df = self.clustered_df.rename(columns={"index": ""})
        
        """ Add batch column """
        batch_str_list = [self.batch_idx2str[batch_idx] for batch_idx in self.batch_idx_list]
        new_col = pd.Series(batch_str_list, name="batch")
        self.clustered_df = pd.concat([self.clustered_df, new_col], axis=1)
        
        """ Add day column """
        new_col = pd.Series(self.day_list, name="day")
        self.clustered_df = pd.concat([self.clustered_df, new_col], axis=1)
        # ---------------------------------------------------------------------/


    def _save_clustered_df(self):
        """ execution dependency:
            - `self.clustered_file`
            - `self.clustered_df`
        """
        basename = os.path.basename(self.clustered_file)
        
        self.clustered_df.to_csv(self.clustered_file, encoding='utf_8_sig', index=False)
        self._cli_out.write(f"{basename} : '{self.clustered_file}'")
        # ---------------------------------------------------------------------/


    def _save_kmeans_centers(self):
        """
        """
        temp_dict = {i: center for i, center in enumerate(self.kmeans_centers.squeeze())}
        save_dict = {}
        
        for k, v in self.cidx2clabel.items():
            save_dict[v] = temp_dict[k]
        
        path = self.dst_root.joinpath("kmeans_centers.toml")
        dump_config(path, save_dict)
        # ---------------------------------------------------------------------/