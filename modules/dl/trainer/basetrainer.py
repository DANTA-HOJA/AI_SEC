import os
import random
import re
import shutil
import sys
import traceback
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import imgaug as ia
import numpy as np
import pandas as pd
import torch
from colorama import Back, Fore, Style
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ...data.dataset.utils import parse_dataset_file_name
from ...data.processeddatainstance import ProcessedDataInstance
from ...shared.baseobject import BaseObject
from ...shared.config import dump_config
from ...shared.timer import Timer
from ...shared.utils import create_new_dir, formatter_padr0
from ..dataset.imgdataset import ImgDataset_v3
from ..utils import (calculate_metrics, gen_class2num_dict,
                     gen_class_counts_dict, set_gpu)
from .utils import (calculate_class_weight, plot_training_trend,
                    rename_training_dir, save_model, save_training_logs)
# -----------------------------------------------------------------------------/


class BaseTrainer(BaseObject):

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
        
        # GPU settings
        self.device: torch.device = set_gpu(self.cuda_idx, self._cli_out)
        self._set_training_reproducibility()
        if self.use_amp: self._set_amp_scaler()
        
        self._set_dataset_df()
        self._set_mapping_attrs()
        self._set_train_valid_df()
        self._set_class_counts_dict()
        self._print_trainingset_informations()
        self._set_dst_root()
        
        """ Preparing DL components """
        self._set_train_set() # abstract function
        self._set_valid_set() # abstract function
        self._set_dataloaders()
        self._set_model() # abstract function
        self._set_loss_fn() # abstract function
        self._set_optimizer() # abstract function
        if self.use_lr_schedular: self._set_lr_scheduler() # abstract function
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        """ [dataset] """
        self.dataset_seed_dir: str = self.config["dataset"]["seed_dir"]
        self.dataset_data: str = self.config["dataset"]["data"]
        self.dataset_palmskin_result: str = self.config["dataset"]["palmskin_result"]
        self.dataset_base_size: str = self.config["dataset"]["base_size"]
        self.dataset_classif_strategy: str = self.config["dataset"]["classif_strategy"]
        self.dataset_file_name: str = self.config["dataset"]["file_name"]
        
        """ [model] """
        self.model_name: str = self.config["model"]["name"]
        self.model_pretrain: str = self.config["model"]["pretrain"]
        
        """ [train_opts.cpu] """
        self.num_workers: int = self.config["train_opts"]["cpu"]["num_workers"]
        
        """ [train_opts.cuda] """
        self.cuda_idx: int = self.config["train_opts"]["cuda"]["index"]
        self.use_amp: bool = self.config["train_opts"]["cuda"]["use_amp"]
        
        """ [train_opts.debug_mode] """
        self.debug_mode: bool = self.config["train_opts"]["debug_mode"]["enable"]
        self.debug_rand_select:int = self.config["train_opts"]["debug_mode"]["rand_select"]
        
        """ [train_opts.data] """
        self.use_hsv: bool = self.config["train_opts"]["data"]["use_hsv"]
        self.forcing_balance: bool = self.config["train_opts"]["data"]["forcing_balance"]
        self.forcing_sample_amount:int = self.config["train_opts"]["data"]["forcing_sample_amount"]
        self.random_crop: bool = self.config["train_opts"]["data"]["random_crop"]
        self.add_bg_class: bool = self.config["train_opts"]["data"]["add_bg_class"]
        self.aug_on_fly: bool = self.config["train_opts"]["data"]["aug_on_fly"]
        
        """ [train_opts] """
        self.epochs: int = self.config["train_opts"]["epochs"]
        self.batch_size: int = self.config["train_opts"]["batch_size"]
        
        """ [train_opts.optimizer] """
        self.lr: float = self.config["train_opts"]["optimizer"]["learning_rate"]
        self.weight_decay: float = self.config["train_opts"]["optimizer"]["weight_decay"]
        
        """ [train_opts.lr_schedular] """
        self.use_lr_schedular: bool = self.config["train_opts"]["lr_schedular"]["enable"]
        self.lr_schedular_step: int = self.config["train_opts"]["lr_schedular"]["step"]
        self.lr_schedular_gamma: float = self.config["train_opts"]["lr_schedular"]["gamma"]
        
        """ [train_opts.earlystop] """
        self.enable_earlystop: bool = self.config["train_opts"]["earlystop"]["enable"]
        self.max_no_improved: int = self.config["train_opts"]["earlystop"]["max_no_improved"]
        
        # if (self.random_crop) and (not self.add_bg_class):
        #     raise AttributeError(f"Can't set `random_crop` = {self.random_crop} "
        #                          f"if `add_bg_class` = {self.add_bg_class}, "
        #                          "random crop may generate a discard image")
        
        if self.debug_mode:
            self.epochs = 10
            self.batch_size = 16
            self._cli_out.write(f"※　: debug mode, force `epochs` = {self.epochs}")
            self._cli_out.write(f"※　: debug mode, force `batch_size` = {self.batch_size}")
        # ---------------------------------------------------------------------/


    def _set_training_reproducibility(self):
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
        
        """ DataLoader """
        self.g: torch.Generator = torch.Generator()
        self.g.manual_seed(0)
        # ---------------------------------------------------------------------/


    @staticmethod
    def _seed_worker(worker_id):
        """ DataLoader reproducibility
            ref: 'https://pytorch.org/docs/stable/notes/randomness.html'
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # ---------------------------------------------------------------------/


    def _set_amp_scaler(self):
        """ A scaler for 'Automatic Mixed Precision (AMP)'
        """
        self.scaler = GradScaler()
        # ---------------------------------------------------------------------/


    def _set_dataset_df(self):
        """
        """
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
        
        src_root = dataset_cropped.joinpath(self.dataset_seed_dir,
                                            self.dataset_data,
                                            self.dataset_palmskin_result,
                                            self.dataset_base_size)
        
        dataset_file: Path = src_root.joinpath(self.dataset_classif_strategy,
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


    def _set_training_df(self): # NOTE: deprecate
        """
        """
        # self.training_df: pd.DataFrame = \
        #         self.dataset_df[(self.dataset_df["dataset"] == "train") & 
        #                              (self.dataset_df["state"] == "preserve")]
        
        # if self.debug_mode:
        #     self.training_df = self.training_df.sample(n=self.debug_rand_select, 
        #                                                replace=False, 
        #                                                random_state=self.rand_seed)
        #     self._cli_out.write(f"Debug mode, reduce to only {len(self.training_df)} images")
        # ---------------------------------------------------------------------/


    def _set_train_valid_df(self): # NOTE: deprecate
        """ Set below attributes
            - `self.train_df`: pd.DataFrame
            - `self.valid_df`: pd.DataFrame
        """
        # self.train_df: pd.DataFrame = pd.DataFrame(columns=self.training_df.columns)
        # self.valid_df: pd.DataFrame = pd.DataFrame(columns=self.training_df.columns)
        
        # for cls in self.num2class_list:

        #     df: pd.DataFrame = self.training_df[(self.training_df["class"] == cls)]
        #     train: pd.DataFrame = df.sample(frac=self.train_ratio, replace=False,
        #                                     random_state=self.rand_seed)
        #     valid: pd.DataFrame = df[~df.index.isin(train.index)]
        #     self.train_df = pd.concat([self.train_df.astype(train.dtypes), train], ignore_index=True)
        #     self.valid_df = pd.concat([self.valid_df.astype(valid.dtypes), valid], ignore_index=True)

        #     train_d = train[(train["cut_section"] == "D")]
        #     train_u = train[(train["cut_section"] == "U")]
        #     valid_d = valid[(valid["cut_section"] == "D")]
        #     valid_u = valid[(valid["cut_section"] == "U")]

        #     self._cli_out.write(f"{cls}: "
        #                         f"train: [ total: {len(train)}, D: {len(train_d)}, U: {len(train_u)} ] "
        #                         f"valid: [ total: {len(valid)}, D: {len(valid_d)}, U: {len(valid_u)} ]")
        # ---------------------------------------------------------------------/


    def _set_train_valid_df(self):
        """ Set below attributes
            >>> self.train_df: pd.DataFrame
            >>> self.valid_df: pd.DataFrame
        """
        self.train_df: pd.DataFrame = \
                self.dataset_df[(self.dataset_df["dataset"] == "train")]
        
        self.valid_df: pd.DataFrame = \
                self.dataset_df[(self.dataset_df["dataset"] == "valid") & 
                                    (self.dataset_df["state"] == "preserve")]
        
        if self.random_crop:
            self.train_df = self.train_df[(self.train_df["image_size"] == "base")]
        else:
            self.train_df = self.train_df[(self.train_df["image_size"] == "crop")]
        
        # if not self.add_bg_class:
        #     self.train_df = self.train_df[(self.train_df["state"] == "preserve")]
        #     self.valid_df = self.valid_df[(self.valid_df["state"] == "preserve")]
        
        # debug: sampleing for faster speed
        if self.debug_mode:
            self.train_df = self.train_df.sample(n=self.debug_rand_select,
                                                 replace=False,
                                                 random_state=self.rand_seed)
            self.valid_df = self.valid_df.sample(n=self.debug_rand_select,
                                                 replace=False,
                                                 random_state=self.rand_seed)
            self._cli_out.write(f"※　: debug mode, reduce to only {self.debug_rand_select} images")
        # ---------------------------------------------------------------------/


    def _set_class_counts_dict(self):
        """ use processed data `clustered_file` to calculate `self.class_counts_dict`
        """
        # instance_desc = re.split("{|}", self.dataset_data)[1]
        # temp_dict = {"data_processed": {"instance_desc": instance_desc}}
        # self._processed_di.parse_config(temp_dict)
        
        # feature_class = parse_dataset_file_name(self.dataset_file_name)["feature_class"]
        # cluster_desc = f"{feature_class}_{self.dataset_classif_strategy}_{self.dataset_seed_dir}"
        # clustered_file = self._processed_di.clustered_files_dict[cluster_desc]
        # df = pd.read_csv(clustered_file, encoding='utf_8_sig')
        
        # self.class_counts_dict: dict[str, int] = \
        #     gen_class_counts_dict(df, self.num2class_list) # 感覺因為 train_df 是 df.sample 抽的所以會和 train_df 差不多
        
        self.class_counts_dict: dict[str, int] = \
            gen_class_counts_dict(self.train_df, self.num2class_list)
        # ---------------------------------------------------------------------/


    def _print_trainingset_informations(self):
        """
        """
        self._cli_out.write(f"train_data ({len(self.train_df)})")
        [self._cli_out.write(f"{i} : image_name = {self.train_df.iloc[i]['image_name']}") for i in range(5)]
        
        self._cli_out.write(f"valid_data ({len(self.valid_df)})")
        [self._cli_out.write(f"{i} : image_name = {self.valid_df.iloc[i]['image_name']}") for i in range(5)]
        
        # temp_dict: Dict[str, int] = gen_class_counts_dict(self.training_df, self.num2class_list)
        # self._cli_out.write(f"class_weight of `self._processed_di.clustered_file` : {calculate_class_weight(self.class_counts_dict)}")
        
        temp_dict = gen_class_counts_dict(self.train_df, self.num2class_list)
        self._cli_out.write(f"class_weight of `self.train_df` : {calculate_class_weight(temp_dict)}")
        
        temp_dict = gen_class_counts_dict(self.valid_df, self.num2class_list)
        self._cli_out.write(f"class_weight of `self.valid_df` : {calculate_class_weight(temp_dict)}")
        # ---------------------------------------------------------------------/


    def _set_dst_root(self):
        """ Set below attributes
            >>> self.time_stamp: str
            >>> self.dst_root: Path
        """
        model_history: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_history")
        self.time_stamp: str = datetime.now().strftime('%Y%m%d_%H_%M_%S')

        self.dst_root: Path = \
            model_history.joinpath(f"Training_{self.time_stamp}")
        # ---------------------------------------------------------------------/


    def _set_train_set(self): # abstract function
        """
        """
        self.train_set: ImgDataset_v3
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def _set_valid_set(self): # abstract function
        """
        """
        self.valid_set: ImgDataset_v3
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def _set_dataloaders(self):
        """ Set below attributes
            >>> self.train_dataloader: DataLoader
            >>> self.valid_dataloader: DataLoader
        """
        if self.num_workers > 0:
            self.train_dataloader: DataLoader = \
                DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                           pin_memory=True, num_workers=self.num_workers,
                           worker_init_fn=self._seed_worker, generator=self.g)
            
            self.valid_dataloader: DataLoader = \
                DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False,
                           pin_memory=True, num_workers=self.num_workers)
            
            self._cli_out.write(f"※　: multiprocess loading, `num_workers` = {self.num_workers}")

        else:
            self.train_dataloader: DataLoader = \
                DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                           pin_memory=True)
            
            self.valid_dataloader: DataLoader = \
                DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False,
                           pin_memory=True)
        
        self._cli_out.write(f"※　: total train batches: {len(self.train_dataloader)}")
        self._cli_out.write(f"※　: total valid batches: {len(self.valid_dataloader)}")
        # ---------------------------------------------------------------------/


    def _set_model(self): # abstract function
        """
        """
        self.model: torch.nn.Module
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def _set_loss_fn(self): # abstract function
        """
        """
        self.loss_fn: torch.nn.modules.loss._Loss
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def _set_optimizer(self): # abstract function
        """
        """
        self.optimizer: torch.optim.Optimizer
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def _set_lr_scheduler(self): # abstract function
        """
        """
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        create_new_dir(self.dst_root)
        dump_config(self.dst_root.joinpath("training_config.toml"), self.config) # save file
        self._save_training_amount_file() # save file
        
        """ Create Timer """
        timer = Timer()
        
        """ Training """
        self._set_training_attrs()
        self._cli_out.divide()
        self.pbar_n_epoch = tqdm(total=self.epochs, desc=f"Epoch ")
        self.pbar_n_train = tqdm(total=len(self.train_dataloader), desc="Train ")
        self.pbar_n_valid = tqdm(total=len(self.valid_dataloader), desc="Valid ")
        try:
            
            timer.start()
            for epoch in range(1, self.epochs+1):
                # Update progress bar description
                self.pbar_n_epoch.desc = f"Epoch {epoch:{formatter_padr0(self.epochs)}} "
                self.pbar_n_epoch.refresh()
                
                self._one_epoch_training(epoch)
                self._one_epoch_validating(epoch)
                
                """ Save logs (convert to Dataframe) """
                save_training_logs(self.dst_root, self.train_logs, self.valid_logs,
                                   self.best_val_log) # save file*2
                
                """ Save plot """
                plot_training_trend_kwargs = {
                    "save_dir"   : self.dst_root,
                    "loss_key"   : "average_loss",
                    "score_key"  : self.score_key,
                    "train_logs" : self.train_logs,
                    "valid_logs" : self.valid_logs,
                }
                plot_training_trend(**plot_training_trend_kwargs) # save file
                
                """ Print `output_string` """
                self._cli_out.write(self.output_string)
                
                """ SystemExit condition """
                if self.accum_no_improved == self.max_no_improved:
                    sys.exit() # raise SystemExit
                
                # Update `pbar_n_epoch`
                self.pbar_n_epoch.update(1)
                self.pbar_n_epoch.refresh()
        
        except KeyboardInterrupt:
            self._close_pbars()
            self.training_state = "KeyboardInterrupt"
            tqdm.write("KeyboardInterrupt")
        
        except SystemExit:
            self._close_pbars()
            self.training_state = "EarlyStop"
            tqdm.write("EarlyStop, exit training")
        
        except Exception as e:
            self._close_pbars()
            self.training_state = "ExceptionError"
            tqdm.write(traceback.format_exc())
            with open(self.dst_root.joinpath(r"{Logs}_ExceptionError.log"), mode="w") as f_writer:
                f_writer.write(traceback.format_exc())

        else:
            self._close_pbars()
            self.training_state = "Completed"
            tqdm.write("Training Completed")
        
        finally:
            """ Save training consume time """
            timer.stop()
            timer.calculate_consume_time()
            timer.save_consume_time(self.dst_root, desc="training time") # save file
            
            if self.best_val_log["epoch"] > 0:
                """ NOTE: Reason of this condition:
                
                    If `best_val_log["epoch"]` > 0,
                    all of `logs` and `state_dict` are not empty.
                """
                
                """ Save model """
                save_model("best", self.dst_root, self.best_model_state_dict, self.best_optimizer_state_dict) # save file
                save_model("final", self.dst_root, self.model.state_dict(), self.optimizer.state_dict()) # save file

                """ Rename `dst_root` """
                # new_name_format : {time_stamp}_{training_state}_{target_epochs_with_ImgLoadOptions}
                # example : '20230920_13_18_51_{EarlyStop}_{120_epochs_AugOnFly}'
                rename_training_dir_kwargs = {
                    "orig_dir"   : self.dst_root,
                    "time_stamp" : self.time_stamp,
                    "state"      : self.training_state,
                    "epochs"     : self.valid_logs[-1]["epoch"],
                    "aug_on_fly" : self.aug_on_fly,
                    "use_hsv"    : self.use_hsv
                }
                rename_training_dir(**rename_training_dir_kwargs)
                
            else:
                """ Delete folder if less than one epoch has been completed. """
                self._cli_out.write(f"Less than One epoch has been completed, "
                                    f"remove directory '{self.dst_root}' ")
                shutil.rmtree(self.dst_root)
            
            self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _save_training_amount_file(self):
        """ Save an empty file but file name is training amount info
        """
        train_num = len(self.train_df)
        valid_num = len(self.valid_df)
        training_num = train_num + valid_num
        
        training_amount = f"{{ dataset_{training_num} }}_"
        training_amount += f"{{ train_{train_num} }}_"
        training_amount += f"{{ valid_{valid_num} }}"
        
        save_path = self.dst_root.joinpath(training_amount)
        with open(save_path, mode="w") as f_writer: pass
        # ---------------------------------------------------------------------/


    def _set_training_attrs(self):
        """
        """
        self.train_logs: List[dict] = []
        self.valid_logs: List[dict] = []
        self.output_string: str = "" # for CLIOutput
        
        """ best record variables """
        self.best_val_f1: float = 0.0
        self.best_val_log: dict = { "Best": self.time_stamp, "epoch": 0 }
        self.best_model_state_dict: dict = deepcopy(self.model.state_dict())
        self.best_optimizer_state_dict: dict = deepcopy(self.optimizer.state_dict())
        
        """ early stop """
        self.best_val_avg_loss: float = np.inf
        self.accum_no_improved: int = 0
        
        """ exception """
        self.training_state: str = ""
        
        # score
        self.score_key: str = "maweavg_f1"
        self._cli_out.write("※　evaluation metric : "
                                "maweavg_f1 = ( macro_f1 + weighted_f1 ) / 2")
        # ---------------------------------------------------------------------/


    def _one_epoch_training(self, epoch:int):
        """
        """
        log: dict = { "Train": "", "epoch": epoch }
        pred_list: list = []
        gt_list: list = []
        accum_loss: float = 0.0
        self.output_string = f"Epoch: {epoch:{formatter_padr0(self.epochs)}}"
        self.pbar_n_train.n = 0
        self.pbar_n_train.refresh()
        
        self.model.train() # set model to training mode
        for data in self.train_dataloader:
            
            images, labels, crop_names = data
            images, labels = images.to(self.device), labels.to(self.device) # move to GPU
            
            self.optimizer.zero_grad() # clean gradients before each backpropagation
            if self.use_amp:
                with autocast():
                    preds = self.model(images)
                    loss_value = self.loss_fn(preds, labels)
                    
                self.scaler.scale(loss_value).backward() # 計算並縮放損失的梯度
                self.scaler.step(self.optimizer) # 更新模型參數
                self.scaler.update() # 更新縮放因子
                
            else:
                preds = self.model(images)
                loss_value = self.loss_fn(preds, labels)
                
                loss_value.backward() # update model_parameters by backpropagation
                self.optimizer.step()
            
            """ Accumulate current batch loss """
            accum_loss += loss_value.item() # tensor.item() -> get value of a Tensor
            
            """ Extend `pred_list`, `gt_list` """
            preds_prob = torch.nn.functional.softmax(preds, dim=1)
            _, preds_hcls = torch.max(preds_prob, 1) # get the highest probability class
            pred_list.extend(preds_hcls.cpu().numpy().tolist()) # conversion flow: Tensor --> ndarray --> list
            gt_list.extend(labels.cpu().numpy().tolist())
            
            """ Update `pbar_n_train` """
            self.pbar_n_train.update(1)
            self.pbar_n_train.refresh()
        
        if self.use_lr_schedular: self.lr_scheduler.step() # update 'lr' for each epoch
        
        calculate_metrics(log, (accum_loss/len(self.train_dataloader)), 
                          pred_list, gt_list, self.class2num_dict)
        
        """ Update `self.train_logs` """
        self.train_logs.append(log)
        
        """ Update postfix of `pbar_n_train` """
        temp_str = f" {'{'} Loss: {log['average_loss']}, "
        temp_str += f"{self.score_key}: {log[self.score_key]}"
        if self.use_lr_schedular is True:
            temp_str += f", lr: {self.lr_scheduler.get_last_lr()[0]:.0e} {'}'} "
        else:
            temp_str += f" {'}'} "
        self.pbar_n_train.postfix = temp_str
        self.pbar_n_train.refresh()
        # ---------------------------------------------------------------------/


    def _one_epoch_validating(self, epoch:int):
        """
        """
        log: dict = { "Valid": "", "epoch": epoch }
        pred_list: list = []
        gt_list: list = []
        accum_loss: float = 0.0
        self.pbar_n_valid.n = 0
        self.pbar_n_valid.refresh()
        
        self.model.eval() # set to evaluation mode
        with torch.no_grad():
            for data in self.valid_dataloader:
                
                images, labels, crop_names = data
                images, labels = images.to(self.device), labels.to(self.device) # move to GPU
                
                preds = self.model(images)
                loss_value = self.loss_fn(preds, labels)
                
                """ Accumulate current batch loss """
                accum_loss += loss_value.item() # tensor.item() -> get value of a Tensor
                
                """ Extend `pred_list`, `gt_list` """
                preds_prob = torch.nn.functional.softmax(preds, dim=1)
                _, preds_hcls = torch.max(preds_prob, 1) # get the highest probability class
                pred_list.extend(preds_hcls.cpu().numpy().tolist()) # conversion flow: Tensor --> ndarray --> list
                gt_list.extend(labels.cpu().numpy().tolist())
                
                """ Update `pbar_n_valid` """
                self.pbar_n_valid.update(1)
                self.pbar_n_valid.refresh()

        calculate_metrics(log, (accum_loss/len(self.valid_dataloader)),
                          pred_list, gt_list, self.class2num_dict)
        
        """ Update `self.valid_logs` """
        self.valid_logs.append(log)
        
        """ Update best record """
        log["best_record"] = ""
        if log[self.score_key] > self.best_val_f1:
            
            log["best_record"] = "☆★☆ BEST_VALIDATION_SCORE ☆★☆"
            self.best_val_f1 = log[self.score_key]
                        
            """ Update `best_val_log` """
            calculate_metrics(self.best_val_log, (accum_loss/len(self.valid_dataloader)),
                              pred_list, gt_list, self.class2num_dict)
            
            self.best_model_state_dict = deepcopy(self.model.state_dict())
            self.best_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
            self.best_val_log["epoch"] = epoch
            
            self.output_string += (f", ☆★☆ BEST_VALIDATION_SCORE ☆★☆"
                                   f", best_val_{self.score_key} = {self.best_val_log[self.score_key]}")
        # ---------------------------------------------------------------------
        
        """ Check 'EarlyStop' """
        log["early_stop"] = ""
        if log["average_loss"] < self.best_val_avg_loss:
            self.best_val_avg_loss = log["average_loss"]
            if self.enable_earlystop: self.accum_no_improved = 0
            
            self.output_string += (f", ☆★☆ BEST_VALID_LOSS ☆★☆"
                                   f", best_valid_avg_loss = {self.best_val_avg_loss}")
        else:
            tmp_string = "◎㊣◎ VALID_LOSS_NO_IMPROVED ◎㊣◎"
            log["early_stop"] = tmp_string
            self.output_string += f", {tmp_string}"
            
            if self.enable_earlystop:
                self.accum_no_improved += 1
                tmp_string = f", accum_no_improved = {self.accum_no_improved}"
                log["early_stop"] += tmp_string
                self.output_string += tmp_string
        # ---------------------------------------------------------------------
        
        """ Update postfix of `pbar_n_valid` """
        temp_str = f" {'{'} Loss: {log['average_loss']}, "
        temp_str += f"{self.score_key}: {log[self.score_key]} {'}'} "
        self.pbar_n_valid.postfix = temp_str
        self.pbar_n_valid.refresh()
        # ---------------------------------------------------------------------/


    def _close_pbars(self):
        """ Close below attributes
            - `self.pbar_n_epoch`
            - `self.pbar_n_train`
            - `self.pbar_n_valid`
        """
        self.pbar_n_epoch.close()
        self.pbar_n_train.close()
        self.pbar_n_valid.close()
        
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/