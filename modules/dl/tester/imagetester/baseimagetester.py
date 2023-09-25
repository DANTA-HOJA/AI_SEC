import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from collections import Counter
from copy import deepcopy

import random
import numpy as np
import pandas as pd
import tomlkit
from tomlkit.toml_document import TOMLDocument
from colorama import Fore, Back, Style
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import imgaug as ia
from sklearn.metrics import classification_report

from ..utils import confusion_matrix_with_class, rename_history_dir
from ...trainer.utils import calculate_class_weight
from ...utils import set_gpu, gen_class2num_dict, gen_class_counts_dict, \
                     test_read_image, calculate_metrics
from ....shared.clioutput import CLIOutput
from ....shared.config import load_config, dump_config
from ....shared.pathnavigator import PathNavigator
from ....shared.utils import formatter_padr0
from ....assert_fn import assert_0_or_1_history_dir
# -----------------------------------------------------------------------------/


class BaseImageTester:


    def __init__(self) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self._path_navigator = PathNavigator()
        self._cli_out = CLIOutput
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.config: Union[dict, TOMLDocument] = load_config(config_file, cli_out=self._cli_out)
        self._set_config_attrs()
        self._set_history_dir()
        self._set_train_config_attrs()
        self._set_dataset_xlsx_path()

        """ Testing reproducibility """
        self._set_testing_reproducibility()
        
        """ Set GPU """
        self.device: torch.device = set_gpu(self.cuda_idx, self._cli_out)
        
        """ Load `dataset_xlsx` """
        self.dataset_xlsx_df: pd.DataFrame = pd.read_excel(self.dataset_xlsx_path, engine='openpyxl')
        self._set_mapping_attrs()
        
        """ Set components' necessary variables """
        self._set_test_df()
        self._print_testset_informations()
        if self.debug_mode:
            test_read_image(Path(self.test_df.iloc[-1]["path"]), self._cli_out)
        
        """ Save files """
        self._save_test_amount_file() # save file
        
        """ Preparing DL components """
        self._set_test_set() # abstract function
        self._set_test_dataloader()
        self._set_model() # abstract function
        self._set_loss_fn() # abstract function
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="3.1.test_by_image.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
        """ Testing """
        self._set_testing_attrs()
        self._cli_out.divide()
        self.pbar_n_test = tqdm(total=len(self.test_dataloader), desc="Test ")
        
        self._one_epoch_testing()
        
        self.pbar_n_test.close()
        self._cli_out.new_line()
        
        """ Save files """
        self._save_test_log(test_desc="PredByImg", score_key="maweavg_f1") # save file
        self._save_report(test_desc="PredByImg") # save file
        
        """ Rename `history_dir` """
        # new_name_format : {time_stamp}_{test_desc}_{target_epochs_with_ImgLoadOptions}_{model_state}_{score_key}
        # example : '20230630_04_39_25_{Tested_PredByImg}_{100_epochs_AugOnFly}_{best}_{maweavg_f1_0.90208}'
        rename_history_dir(self.history_dir, "Tested_PredByImg",
                           self.model_state, self.test_log, score_key="maweavg_f1")
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
        if self.model_state not in ["best", "final"]:
            raise ValueError(f"config: `model_prediction.state`: "
                             f"'{self.model_state}', accept 'best' or 'final' only\n")
        
        model_prediction: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_prediction")
        
        best_found = []
        final_found = []
        found_list = list(model_prediction.glob(f"{self.model_time_stamp}*"))
        for i, path in enumerate(found_list):
            if f"{{best}}" in str(path): best_found.append(found_list.pop(i))
            if f"{{final}}" in str(path): final_found.append(found_list.pop(i))

        if self.model_state == "best" and best_found:
            assert_0_or_1_history_dir(best_found, self.model_time_stamp, self.model_state)
            self.history_dir = best_found[0]
            return
        
        if self.model_state == "final" and final_found:
            assert_0_or_1_history_dir(final_found, self.model_time_stamp, self.model_state)
            self.history_dir = final_found[0]
            return
        
        assert_0_or_1_history_dir(found_list, self.model_time_stamp, self.model_state)
        if found_list:
            self.history_dir = found_list[0]
            return
        else:
            raise ValueError("No `history_dir` matches the provided config")
        # ---------------------------------------------------------------------/



    def _set_train_config_attrs(self):
        """
        """
        path = self.history_dir.joinpath("train_config.toml")
        if not path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find `dataset_xlsx`"
                                    f"run one of the training scripts in `2.training` directory. "
                                    f"{Style.RESET_ALL}\n")
        
        self.train_config: Union[dict, TOMLDocument] = load_config(path, cli_out=self._cli_out)
        
        """ [dataset] """
        self.dataset_seed_dir: str = self.train_config["dataset"]["seed_dir"]
        self.dataset_name: str = self.train_config["dataset"]["name"]
        self.dataset_result_alias: str = self.train_config["dataset"]["result_alias"]
        self.dataset_classif_strategy: str = self.train_config["dataset"]["classif_strategy"]
        self.dataset_xlsx_name: str = self.train_config["dataset"]["xlsx_name"]
        
        """ [model] """
        self.model_name: str = self.train_config["model"]["name"]
        
        """ [train_opts] """
        self.rand_seed: int = self.train_config["train_opts"]["random_seed"]
        
        """ [train_opts.data] """
        self.use_hsv: bool = self.train_config["train_opts"]["data"]["use_hsv"]
        # ---------------------------------------------------------------------/



    def _set_dataset_xlsx_path(self):
        """
        """
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v2")
        
        self.dataset_xlsx_path: Path = dataset_cropped.joinpath(self.dataset_seed_dir,
                                                                self.dataset_name,
                                                                self.dataset_result_alias,
                                                                self.dataset_classif_strategy,
                                                                f"{self.dataset_xlsx_name}.xlsx")
        if not self.dataset_xlsx_path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find `dataset_xlsx`"
                                    f"run `1.3.create_dataset_xlsx.py` before training. "
                                    f"{Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def _set_testing_reproducibility(self):
        """ Pytorch reproducibility
            - ref: https://clay-atlas.com/us/blog/2021/08/24/pytorch-en-set-seed-reproduce/?amp=1
            - ref: https://pytorch.org/docs/stable/notes/randomness.html
        """
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



    def _set_mapping_attrs(self):
        """ Set below attributes
            - `self.num2class_list`: list
            - `self.class2num_dict`: Dict[str, int]
            
        example output :
        >>> num2class_list = ['L', 'M', 'S'], class2num_dict = {'L': 0, 'M': 1, 'S': 2}
        """
        self.num2class_list: list = sorted(Counter(self.dataset_xlsx_df["class"]).keys())
        self.class2num_dict: Dict[str, int] = gen_class2num_dict(self.num2class_list)
        
        self._cli_out.write(f"num2class_list = {self.num2class_list}, "
                            f"class2num_dict = {self.class2num_dict}")
        # ---------------------------------------------------------------------/



    def _set_test_df(self):
        """
        """
        self.test_df: pd.DataFrame = \
                self.dataset_xlsx_df[(self.dataset_xlsx_df["dataset"] == "test") & 
                                     (self.dataset_xlsx_df["state"] == "preserve")]
        
        if self.debug_mode:
            self.test_df = self.test_df.sample(n=self.debug_rand_select, 
                                                     replace=False, 
                                                     random_state=self.rand_seed)
            self._cli_out.write(f"Debug mode, reduce to only {len(self.test_df)} images")
        # ---------------------------------------------------------------------/



    def _print_testset_informations(self):
        """
        """
        for cls in self.num2class_list:
            test: pd.DataFrame = self.test_df[(self.test_df["class"] == cls)]
            test_d = test[(test["cut_section"] == "D")]
            test_u = test[(test["cut_section"] == "U")]
            self._cli_out.write(f"{cls}: "
                                f"test: [ total: {len(test)}, D: {len(test_d)}, U: {len(test_u)} ]")
        
        self._cli_out.write(f"test_data ({len(self.test_df)})")
        [self._cli_out.write(f"{i} : image_name = {self.test_df.iloc[i]['image_name']}") for i in range(5)]
        
        temp_dict: Dict[str, int] = gen_class_counts_dict(self.test_df, self.num2class_list)
        self._cli_out.write(f"class_weight of `self.test_df` : {calculate_class_weight(temp_dict)}")
        # ---------------------------------------------------------------------/



    def _save_test_amount_file(self):
        """ Save an empty file but file name is test amount info
        """
        test_amount = f"{{ datatest_{len(self.test_df)} }}_"
        test_amount += f"{{ test_{len(self.test_df)} }}"
        
        save_path = self.history_dir.joinpath(test_amount)
        with open(save_path, mode="w") as f_writer: pass
        # ---------------------------------------------------------------------/



    def _set_test_set(self): # abstract function
        """
        """
        self.test_set: Dataset
        
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



    def _set_testing_attrs(self):
        """
        """
        self.test_log: dict = {}
        self.pred_list_to_name: List[str] = []
        self.gt_list_to_name: List[str] = []
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
                
                images, labels, crop_names = data
                images, labels = images.to(self.device), labels.to(self.device) # move to GPU
                
                preds = self.model(images)
                loss_value = self.loss_fn(preds, labels)
                
                """ Extend `pred_list`, `gt_list` """
                preds_prob = torch.nn.functional.softmax(preds, dim=1)
                _, preds_hcls = torch.max(preds_prob, 1) # get the highest probability class
                pred_list.extend(preds_hcls.cpu().numpy().tolist()) # conversion flow: Tensor --> ndarray --> list
                gt_list.extend(labels.cpu().numpy().tolist())
                
                """ Add current batch loss """
                accum_loss += loss_value.item() # get value of a Tensor
                
                """ Print number of matches in current batch """
                num_match = (preds_hcls.cpu() == labels.cpu()).sum().item()
                self._cli_out.write(f"Batch[ {(batch+1):{formatter_padr0(self.test_dataloader)}} / {len(self.test_dataloader)} ], "
                                    f"# of (ground truth == prediction) in this batch : "
                                    f"{num_match:{formatter_padr0(labels)}} / {len(labels)} "
                                    f"( {num_match/len(labels):.2f} )")
                
                """ Update `pbar_n_valid` """
                self.pbar_n_test.update(1)
                self.pbar_n_test.refresh()

        calculate_metrics(self.test_log, (accum_loss/len(self.test_dataloader)),
                          pred_list, gt_list, self.class2num_dict)
        
        self.pred_list_to_name = [ self.num2class_list[i] for i in pred_list ]
        self.gt_list_to_name = [ self.num2class_list[i] for i in gt_list ]
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
            
            config_in_report = deepcopy(self.train_config["dataset"])
            f_writer.write("[ dataset ]\n")
            f_writer.write(f"{tomlkit.dumps(config_in_report)}\n")
            
            f_writer.write(f"※ For more detail info please refer to its 'train_config' file...\n\n\n")


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