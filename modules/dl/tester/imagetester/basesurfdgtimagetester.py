import json
import os
import random
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import imgaug as ia
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tomlkit
import torch
from colorama import Back, Fore, Style
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from tomlkit.toml_document import TOMLDocument
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ....data.processeddatainstance import ProcessedDataInstance
from ....shared.baseobject import BaseObject
from ....shared.config import dump_config, load_config
from ....shared.utils import formatter_padr0
from ...dataset.imgdataset import SurfDGTImgDataset_v3
from ...tester.utils import get_history_dir
from ...trainer.utils import calculate_class_weight
from ...utils import (calculate_metrics, calculate_r_squared,
                      gen_class2num_dict, gen_class_counts_dict, set_gpu)
from ..utils import confusion_matrix_with_class, rename_history_dir
# -----------------------------------------------------------------------------/


class BaseSurfDGTImageTester(BaseObject):

    def __init__(self, processed_data_instance:ProcessedDataInstance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        
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
        self._set_history_dir()
        self._set_training_config_attrs()
        
        # GPU settings
        self.device: torch.device = set_gpu(self.cuda_idx, self._cli_out)
        self._set_testing_reproducibility()
        
        self._set_dataset_df()
        self._set_mapping_attrs()
        self._set_test_df()
        self._print_testset_informations()
        self._set_kmeans_attrs()
        
        """ Preparing DL components """
        self._set_test_set() # abstract function
        self._set_test_dataloader()
        self._set_model() # abstract function
        self._set_loss_fn() # abstract function
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        """ [model_prediction] """
        self.model_time_stamp: str = self.config["model_prediction"]["time_stamp"]
        self.model_state: str = self.config["model_prediction"]["state"]
        
        """ [test_opts] """
        self.batch_size: int = self.config["test_opts"]["batch_size"]
        
        """ [test_opts.debug_mode] """
        self.debug_mode: bool = self.config["test_opts"]["debug_mode"]["enable"]
        self.debug_rand_select:int = self.config["test_opts"]["debug_mode"]["rand_select"]
        
        """ [test_opts.cuda] """
        self.cuda_idx: int = self.config["test_opts"]["cuda"]["index"]
        # ---------------------------------------------------------------------/


    def _set_history_dir(self):
        """
        """
        self.history_dir = get_history_dir(self._path_navigator,
                                           self.model_time_stamp,
                                           self.model_state,
                                           cli_out=self._cli_out)
        # ---------------------------------------------------------------------/


    def _set_training_config_attrs(self):
        """
        """
        path = self.history_dir.joinpath("training_config.toml")
        if not path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} "
                                    f"Can't find 'training_config.toml' "
                                    f"( loss the most important file ). "
                                    f"{Style.RESET_ALL}\n")
        
        self.training_config: Union[dict, TOMLDocument] = \
                                load_config(path, cli_out=self._cli_out)
        
        """ [dataset] """
        self.dataset_seed_dir: str = self.training_config["dataset"]["seed_dir"]
        self.dataset_data: str = self.training_config["dataset"]["data"]
        self.dataset_palmskin_result: str = self.training_config["dataset"]["palmskin_result"]
        self.dataset_base_size: str = self.training_config["dataset"]["base_size"]
        self.dataset_classif_strategy: str = self.training_config["dataset"]["classif_strategy"]
        self.dataset_file_name: str = self.training_config["dataset"]["file_name"]
        
        """ [model] """
        self.model_name: str = self.training_config["model"]["name"]
        
        """ [train_opts.data] """
        self.add_bg_class: bool = self.training_config["train_opts"]["data"]["add_bg_class"]
        # ---------------------------------------------------------------------/


    def _set_testing_reproducibility(self):
        """ Pytorch reproducibility
            - ref: https://clay-atlas.com/us/blog/2021/08/24/pytorch-en-set-seed-reproduce/?amp=1
            - ref: https://pytorch.org/docs/stable/notes/randomness.html
        
        Set below attributes
        >>> self.rand_seed: int
        """
        self.rand_seed: int = int(self.dataset_seed_dir.replace("RND", ""))
        
        """ Seeds """
        random.seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        torch.cuda.manual_seed(self.rand_seed) # current GPU
        torch.cuda.manual_seed_all(self.rand_seed) # all GPUs
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        ia.seed(self.rand_seed)
        # ---------------------------------------------------------------------/


    def _set_dataset_df(self):
        """
        """
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
        
        self.src_root = dataset_cropped.joinpath(self.dataset_seed_dir,
                                            self.dataset_data,
                                            self.dataset_palmskin_result,
                                            self.dataset_base_size)
        
        dataset_file: Path = self.src_root.joinpath(self.dataset_classif_strategy,
                                               self.dataset_file_name)
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find target dataset file "
                                    f"run `1.2.create_dataset_file.py` to create it. "
                                    f"{Style.RESET_ALL}\n")
        
        self.dataset_df: pd.DataFrame = \
            pd.read_csv(dataset_file, encoding='utf_8_sig')
        # ---------------------------------------------------------------------/


    def _set_mapping_attrs(self):
        """ Set below attributes
            >>> self.num2class_list: list
            >>> self.class2num_dict: Dict[str, int]
        
        Example :
        >>> num2class_list = ['L', 'M', 'S']
        >>> class2num_dict = {'L': 0, 'M': 1, 'S': 2}
        """
        cls_list = list(Counter(self.dataset_df["class"]).keys())
        
        if self.add_bg_class:
            cls_list.append("BG")
        
        self.num2class_list: list = sorted(cls_list)
        self.class2num_dict: Dict[str, int] = gen_class2num_dict(self.num2class_list)
        
        self._cli_out.write(f"num2class_list = {self.num2class_list}, "
                            f"class2num_dict = {self.class2num_dict}")
        # ---------------------------------------------------------------------/


    def _set_test_df(self):
        """
        """
        self.test_df: pd.DataFrame = \
                self.dataset_df[(self.dataset_df["dataset"] == "test") &
                                    (self.dataset_df["state"] == "preserve")]
        
        # if not self.add_bg_class:
        #     self.test_df = self.test_df[(self.test_df["state"] == "preserve")]
        
        # debug: sampleing for faster speed
        if self.debug_mode:
            self.test_df = self.test_df.sample(n=self.debug_rand_select,
                                               replace=False,
                                               random_state=self.rand_seed)
            self._cli_out.write(f"※　: debug mode, reduce to only {self.debug_rand_select} images")
        # ---------------------------------------------------------------------/


    def _print_testset_informations(self):
        """
        """
        # for cls in self.num2class_list:
        #     test: pd.DataFrame = self.test_df[(self.test_df["class"] == cls)]
        #     test_d = test[(test["cut_section"] == "D")]
        #     test_u = test[(test["cut_section"] == "U")]
        #     self._cli_out.write(f"{cls}: "
        #                         f"test: [ total: {len(test)}, D: {len(test_d)}, U: {len(test_u)} ]")
        
        self._cli_out.write(f"test_data ({len(self.test_df)})")
        [self._cli_out.write(f"{i} : image_name = {self.test_df.iloc[i]['image_name']}") for i in range(5)]
        
        # temp_dict: Dict[str, int] = gen_class_counts_dict(self.test_df, self.num2class_list)
        # self._cli_out.write(f"class_weight of `self.test_df` : {calculate_class_weight(temp_dict)}")
        # ---------------------------------------------------------------------/


    def _set_kmeans_attrs(self):
        """
        """
        temp_dict = {"data_processed": {"instance_desc": re.split("{|}", self.dataset_data)[1]}}
        self._processed_di.parse_config(temp_dict)

        feature_class = f"SURF{len(Counter(self.test_df['class']))}C"
        cluster_desc = f"{feature_class}_{self.dataset_classif_strategy}_{self.dataset_seed_dir}"
        
        # kmeans_model
        path = self._processed_di.clustered_file_dir.joinpath(cluster_desc, "kmeans_model.joblib")
        self.kmeans: KMeans = joblib.load(path)
        
        # kmeans_mapping
        path = self._processed_di.clustered_file_dir.joinpath(cluster_desc, "kmeans_mapping.json")
        with open(path, mode="r") as f_reader:
            self.kmeans_mapping: dict[int, str] = json.load(f_reader)
        # ---------------------------------------------------------------------/


    def _set_test_set(self): # abstract function
        """
        """
        self.test_set: SurfDGTImgDataset_v3
        
        raise NotImplementedError("This is a base image tester, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def _set_test_dataloader(self):
        """
        """
        self.test_dataloader: DataLoader = \
            DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                       pin_memory=True)
        
        self._cli_out.write(f"※　: total test batches: {len(self.test_dataloader)}")
        # ---------------------------------------------------------------------/


    def _set_model(self): # abstract function
        """
        """
        self.model: torch.nn.Module
        
        raise NotImplementedError("This is a base image tester, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def _set_loss_fn(self): # abstract function
        """
        """
        self.loss_fn: torch.nn.modules.loss._Loss
        
        raise NotImplementedError("This is a base image tester, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super(BaseSurfDGTImageTester, self).run(config)
        
        self._save_testing_amount_file() # save file
        
        """ Testing """
        self._set_testing_attrs()
        self._cli_out.divide()
        self.pbar_n_test = tqdm(total=len(self.test_dataloader),
                                desc="Test (PredByImg) ")
        
        self._one_epoch_testing()
        
        self.pbar_n_test.close()
        self._cli_out.new_line()
        
        """ Save files """
        self._save_test_log(test_desc="PredByImg", score_key="r_squared") # save file
        self._save_report(test_desc="PredByImg") # save file
        self._save_r2_figure()
        
        """ Rename `history_dir` """
        # new_name_format : {time_stamp}_{test_desc}_{target_epochs_with_ImgLoadOptions}_{model_state}_{score_key}
        # example : '20230630_04_39_25_{Tested_PredByImg}_{100_epochs_AugOnFly}_{best}_{maweavg_f1_0.90208}'
        rename_history_dir(self.history_dir, "Tested_PredByImg",
                           self.model_state, self.test_log, score_key="r_squared",
                           cli_out=self._cli_out)
        
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _save_testing_amount_file(self):
        """ Save an empty file but file name is test amount info
        """
        test_amount = f"{{ datatest_{len(self.test_df)} }}_"
        test_amount += f"{{ test_{len(self.test_df)} }}"
        
        save_path = self.history_dir.joinpath(test_amount)
        with open(save_path, mode="w") as f_writer: pass
        # ---------------------------------------------------------------------/


    def _set_testing_attrs(self):
        """
        """
        self.test_log: dict = {}
        self.pred_list_to_ori_scale: List[float] = []
        self.pred_list_to_name: List[str] = []
        self.gt_list_to_name: List[str] = list(self.test_df["class"])
        # ---------------------------------------------------------------------/


    def _one_epoch_testing(self):
        """
        """
        pred_list: list = []
        gt_list: list = []
        accum_loss: float = 0.0
        
        self.model.eval() # set to evaluation mode
        with torch.no_grad():
            for batch, data in enumerate(self.test_dataloader):
                
                images, _, areas, crop_names = data
                areas = areas.unsqueeze(1)
                images, areas = images.to(self.device), areas.to(self.device) # move to GPU
                
                preds = self.model(images)
                loss_value = self.loss_fn(preds, areas)
                
                """ Accumulate current batch loss """
                accum_loss += loss_value.item() # tensor.item() -> get value of a Tensor
                
                """ Extend `pred_list`, `gt_list` """
                # preds_prob = torch.nn.functional.softmax(preds, dim=1)
                # _, preds_hcls = torch.max(preds_prob, 1) # get the highest probability class
                pred_list.extend(preds.cpu().numpy().squeeze().tolist()) # conversion flow: Tensor --> ndarray --> list
                gt_list.extend(areas.cpu().numpy().squeeze().tolist())
                
                """ Print number of matches in current batch """
                # num_match = (preds_hcls.cpu() == labels.cpu()).sum().item()
                # self._cli_out.write(f"Batch[ {(batch+1):{formatter_padr0(self.test_dataloader)}} / {len(self.test_dataloader)} ], "
                #                     f"# of (ground truth == prediction) in this batch : "
                #                     f"{num_match:{formatter_padr0(labels)}} / {len(labels)} "
                #                     f"( {num_match/len(labels):.2f} )")
                
                """ Update `pbar_n_test` """
                self.pbar_n_test.update(1)
                self.pbar_n_test.refresh()
        
        self.pred_list_to_ori_scale = [pred*self.test_set.scaler for pred in pred_list]
        
        calculate_r_squared(self.test_log, (accum_loss/len(self.test_dataloader)),
                            self.pred_list_to_ori_scale, list(self.test_df["area"]))
        
        # apply KMeans to predicted surface area
        self.pred_list_to_name = self.kmeans.predict(np.array(self.pred_list_to_ori_scale)[:, None])
        self.pred_list_to_name = \
            [self.kmeans_mapping[str(cidx)] for cidx in self.pred_list_to_name.squeeze()]
        
        calculate_metrics(self.test_log, (accum_loss/len(self.test_dataloader)),
                          self.pred_list_to_name, self.gt_list_to_name, self.class2num_dict)
        # ---------------------------------------------------------------------/


    def _save_test_log(self, test_desc:str, score_key:str):
        """
        """
        found_list = list(self.history_dir.glob(f"{{Logs}}_{test_desc}_{score_key}_*"))
        for path in found_list: os.remove(path)
        
        file_name = f"{{Logs}}_{test_desc}_{score_key}_{self.test_log[f'{score_key}']}.toml"
        path = self.history_dir.joinpath(file_name)
        dump_config(path, self.test_log)
        # ---------------------------------------------------------------------/


    def _save_r2_figure(self):
        """
        """
        fig, ax = plt.subplots(1, 1, figsize=(9,9), dpi=200)
        fig.suptitle(f"R-squared ({self.test_log['r_squared']})", fontsize=20)
        
        sns.regplot(x="groundtruth (um^2)", y="prediction (um^2)", ax=ax, ci=None,
                    data=pd.DataFrame({"groundtruth (um^2)": list(self.test_df["area"]),
                                       "prediction (um^2)": self.pred_list_to_ori_scale}))
        
        fig.tight_layout()
        fig.savefig(self.history_dir.joinpath("r_squared.png"))
        plt.close(fig)
        # ---------------------------------------------------------------------/


    def _save_report(self, test_desc:str):
        """
        """
        file_name = f"{{Report}}_{test_desc}.log"
        path = self.history_dir.joinpath(file_name)
        with open(path, mode="w") as f_writer:
            
            """ Write `config` """
            config_in_report = deepcopy(self.config["model_prediction"])
            f_writer.write("[ model_prediction ]\n")
            f_writer.write(f"{tomlkit.dumps(config_in_report)}\n")
            
            config_in_report = deepcopy(self.training_config["dataset"])
            f_writer.write("[ dataset ]\n")
            f_writer.write(f"{tomlkit.dumps(config_in_report)}\n")
            
            f_writer.write(f"※ For more detail info please refer to its 'training_config' file...\n\n\n")
            
            """ Write `classification_report` """
            cls_report = classification_report(y_true=self.gt_list_to_name,
                                               y_pred=self.pred_list_to_name, digits=5)
            f_writer.write("Classification Report:\n\n")
            f_writer.write(f"{cls_report}\n\n")
            
            """ Write `confusion_matrix` """
            _, confusion_matrix = confusion_matrix_with_class(prediction=self.pred_list_to_name,
                                                              ground_truth=self.gt_list_to_name)
            f_writer.write(f"{confusion_matrix}\n")
        # ---------------------------------------------------------------------/