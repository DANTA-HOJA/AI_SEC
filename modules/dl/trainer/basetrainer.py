import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from collections import Counter
from datetime import datetime
import traceback
import shutil

import random
import numpy as np
import pandas as pd
from tomlkit.toml_document import TOMLDocument
from colorama import Fore, Back, Style
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import imgaug as ia

from .utils import plot_training_trend, save_training_logs, \
                   save_model, rename_training_dir
from ..utils import set_gpu, test_read_image, \
                    gen_class2num_dict, calculate_metrics
from ...plot.plt_show import plot_in_rgb
from ...shared.clioutput import CLIOutput
from ...shared.config import load_config, dump_config
from ...shared.pathnavigator import PathNavigator
from ...shared.timer import Timer
from ...shared.utils import create_new_dir, formatter_padr0
# -----------------------------------------------------------------------------/


class BaseTrainer:


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
        self._set_save_dir()
        self._set_dataset_xlsx_path()
        
        """ Training reproducibility """
        self._set_training_reproducibility()
        
        """ Set GPU """
        self.device: torch.device = set_gpu(self.cuda_idx, self._cli_out)
        
        """ Load `dataset_xlsx` """
        self.dataset_xlsx_df: pd.DataFrame = pd.read_excel(self.dataset_xlsx_path, engine='openpyxl')
        self._set_mapping_attrs()
        
        """ Set components' necessary variables """
        self._set_training_df()
        self._set_class_counts_dict()
        self._set_train_valid_df()
        if self.debug_mode:
            test_read_image(Path(self.train_df.iloc[-1]["path"]), self._cli_out)
        
        """ Save files """
        create_new_dir(self.save_dir)
        dump_config(self.save_dir.joinpath("train_config.toml"), self.config) # save file
        self._save_training_amount_file() # save file
        
        """ Preparing DL components """
        self._set_train_set()
        self._set_valid_set()
        self._set_dataloaders()
        self._set_model()
        self._set_loss_fn()
        self._set_optimizer()
        if self.use_lr_schedular: self._set_lr_scheduler()
        if self.use_amp: self._set_amp_scaler()
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="2.training.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
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
                
                plot_training_trend_kwargs = {
                    "save_dir"   : self.save_dir,
                    "loss_key"   : "average_loss",
                    "score_key"  : self.score_key,
                    "train_logs" : self.train_logs,
                    "valid_logs" : self.valid_logs,
                }
                plot_training_trend(**plot_training_trend_kwargs)
                
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
            with open(self.save_dir.joinpath(r"{Logs}_ExceptionError.log"), mode="w") as f_writer:
                f_writer.write(traceback.format_exc())

        else:
            self._close_pbars()
            self.training_state = "Completed"
            tqdm.write("Training Completed")
        
        finally:
            """ Save training consume time """
            timer.stop()
            timer.calculate_consume_time()
            timer.save_consume_time(self.save_dir, desc="training time")
            
            if self.best_val_log["epoch"] > 0:
                """ If `best_val_log["epoch"]` > 0, all of `logs` and `state_dict` are not empty. """
                
                """ Save logs (convert to Dataframe) """
                save_training_logs(self.save_dir, self.train_logs, self.valid_logs, self.best_val_log)
                
                """ Save model """
                save_model("best", self.save_dir, self.best_model_state_dict, self.best_optimizer_state_dict)
                save_model("final", self.save_dir, self.model.state_dict(), self.optimizer.state_dict())

                """ Rename `save_dir` """
                # new_name_format = {time_stamp}_{state}_{target_epochs_with_ImgLoadOptions}
                # state = {EarlyStop, Interrupt, Completed, Tested, etc.}
                rename_training_dir_kwargs = {
                    "orig_dir"   : self.save_dir,
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
                                    f"remove directory '{self.save_dir}' ")
                shutil.rmtree(self.save_dir)
        # ---------------------------------------------------------------------/



    def _set_config_attrs(self):
        """
        """
        """ [dataset] """
        self.dataset_seed_dir: str = self.config["dataset"]["seed_dir"]
        self.dataset_name: str = self.config["dataset"]["name"]
        self.dataset_result_alias: str = self.config["dataset"]["result_alias"]
        self.dataset_classif_strategy: str = self.config["dataset"]["classif_strategy"]
        self.dataset_xlsx_name: str = self.config["dataset"]["xlsx_name"]
        
        """ [model] """
        self.model_name: str = self.config["model"]["name"]
        self.model_pretrain: str = self.config["model"]["pretrain"]
        
        """ [train_opts] """
        self.train_ratio: float = self.config["train_opts"]["train_ratio"]
        self.rand_seed: int = self.config["train_opts"]["random_seed"]
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
        
        """ [train_opts.data] """
        self.use_hsv: bool = self.config["train_opts"]["data"]["use_hsv"]
        self.aug_on_fly: bool = self.config["train_opts"]["data"]["aug_on_fly"]
        self.forcing_balance: bool = self.config["train_opts"]["data"]["forcing_balance"]
        self.forcing_sample_amount:int = self.config["train_opts"]["data"]["forcing_sample_amount"]
        
        """ [train_opts.debug_mode] """
        self.debug_mode: bool = self.config["train_opts"]["debug_mode"]["enable"]
        self.debug_rand_select:int = self.config["train_opts"]["debug_mode"]["rand_select"]
        
        """ [train_opts.cuda] """
        self.cuda_idx: int = self.config["train_opts"]["cuda"]["index"]
        self.use_amp: bool = self.config["train_opts"]["cuda"]["use_amp"]
        
        """ [train_opts.cpu] """
        self.num_workers: int = self.config["train_opts"]["cpu"]["num_workers"]
        # ---------------------------------------------------------------------/



    def _set_save_dir(self):
        """
        """
        model_cmd: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_cmd")
        self.time_stamp: str = datetime.now().strftime('%Y%m%d_%H_%M_%S')

        self.save_dir: Path = model_cmd.joinpath(f"Training_{self.time_stamp}")
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



    def _set_training_reproducibility(self):
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



    def _set_training_df(self):
        """
        """
        self.training_df: pd.DataFrame = \
                self.dataset_xlsx_df[(self.dataset_xlsx_df["dataset"] == "train") & 
                                     (self.dataset_xlsx_df["state"] == "preserve")]
        
        if self.debug_mode:
            self.training_df = self.training_df.sample(n=self.debug_rand_select, 
                                                       replace=False, 
                                                       random_state=self.rand_seed)
            self._cli_out.write(f"Debug mode, reduce to only {len(self.training_df)} images")
        # ---------------------------------------------------------------------/



    def _set_class_counts_dict(self):
        """
        """
        counter = Counter(self.training_df["class"])
        self.class_counts_dict: Dict[str, int] = {}
        
        for cls in self.num2class_list:
            self.class_counts_dict[cls] = counter[cls]
        # ---------------------------------------------------------------------/



    def _set_train_valid_df(self):
        """ Set below attributes
            - `self.train_df`: pd.DataFrame
            - `self.valid_df`: pd.DataFrame
        """
        self.train_df: pd.DataFrame = pd.DataFrame(columns=self.training_df.columns)
        self.valid_df: pd.DataFrame = pd.DataFrame(columns=self.training_df.columns)
        
        for cls in self.num2class_list:

            df: pd.DataFrame = self.training_df[(self.training_df["class"] == cls)]
            train: pd.DataFrame = df.sample(frac=self.train_ratio, replace=False,
                                            random_state=self.rand_seed)
            valid: pd.DataFrame = df[~df.index.isin(train.index)]
            self.train_df = pd.concat([self.train_df, train], ignore_index=True)
            self.valid_df = pd.concat([self.valid_df, valid], ignore_index=True)

            train_d = train[(train["cut_section"] == "D")]
            train_u = train[(train["cut_section"] == "U")]
            valid_d = valid[(valid["cut_section"] == "D")]
            valid_u = valid[(valid["cut_section"] == "U")]

            self._cli_out.write(f"{cls}: "
                                f"train: [ total: {len(train)}, D: {len(train_d)}, U: {len(train_u)} ] "
                                f"valid: [ total: {len(valid)}, D: {len(valid_d)}, U: {len(valid_u)} ]")
            
        self._cli_out.write(f"train_data ({len(self.train_df)})")
        [self._cli_out.write(f"{i} : image_name = {self.train_df.iloc[i]['image_name']}") for i in range(5)]
        
        self._cli_out.write(f"valid_data ({len(self.valid_df)})")
        [self._cli_out.write(f"{i} : img_path = {self.valid_df.iloc[i]['image_name']}") for i in range(5)]
        # ---------------------------------------------------------------------/



    def _save_training_amount_file(self):
        """ Save an empty file but file name is training amount info
        """
        training_amount = f"{{ dataset_{len(self.training_df)} }}_"
        training_amount += f"{{ train_{len(self.train_df)} }}_"
        training_amount += f"{{ valid_{len(self.valid_df)} }}"
        
        save_path = self.save_dir.joinpath(training_amount)
        with open(save_path, mode="w") as f_writer: pass
        # ---------------------------------------------------------------------/



    def _set_train_set(self):
        """
        """
        self.train_set: Dataset
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/



    def _set_valid_set(self):
        """
        """
        self.valid_set: Dataset
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/



    def _set_dataloaders(self):
        """ Set below attributes
            - `self.train_dataloader`: DataLoader
            - `self.valid_dataloader`: DataLoader
        """
        if self.num_workers > 0:
            self.train_dataloader: DataLoader = \
                DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                           pin_memory=True, num_workers=self.num_workers,
                           worker_init_fn=self._seed_worker, generator=self.g)
            
            self.valid_dataloader: DataLoader = \
                DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False,
                           pin_memory=True, num_workers=self.num_workers)
            
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



    def _set_model(self):
        """
        """
        self.model: torch.nn.Module
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/



    def _set_loss_fn(self):
        """
        """
        self.loss_fn: torch.nn.modules.loss._Loss
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/



    def _set_optimizer(self):
        """
        """
        self.optimizer: torch.optim.Optimizer
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/



    def _set_lr_scheduler(self):
        """
        """
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler
        
        raise NotImplementedError("This is a base trainer, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/



    def _set_amp_scaler(self):
        """ A scaler for 'Automatic Mixed Precision (AMP)'
        """
        self.scaler = GradScaler()
        # ---------------------------------------------------------------------/



    def _set_training_attrs(self):
        """
        """
        self.train_logs: List[dict] = []
        self.valid_logs: List[dict] = []
        
        """ best record variables """
        self.best_val_log: dict = { "Best": self.time_stamp, "epoch": 0 }
        self.best_val_avg_loss: float = np.inf
        self.best_val_f1: float = 0.0
        self.best_model_state_dict: dict = deepcopy(self.model.state_dict())
        self.best_optimizer_state_dict: dict = deepcopy(self.optimizer.state_dict())
        
        """ early stop """
        self.accum_no_improved: int = 0
        
        """ exception """
        self.training_state: str = ""
        
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
            
            """ Extend `pred_list`, `gt_list` """
            preds_prob = torch.nn.functional.softmax(preds, dim=1)
            _, preds_hcls = torch.max(preds_prob, 1) # get the highest probability class
            pred_list.extend(preds_hcls.cpu().numpy().tolist()) # conversion flow: Tensor --> ndarray --> list
            gt_list.extend(labels.cpu().numpy().tolist())
            
            """ Add current batch loss """
            accum_loss += loss_value.item() # get value of a Tensor
            
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
        accum_loss = 0.0
        output_string = f"Epoch: {epoch:{formatter_padr0(self.epochs)}}"
        self.pbar_n_valid.n = 0
        self.pbar_n_valid.refresh()
        
        self.model.eval() # set to evaluation mode
        with torch.no_grad():
            for data in self.valid_dataloader:
                
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
                
                """ Update `pbar_n_valid` """
                self.pbar_n_valid.update(1)
                self.pbar_n_valid.refresh()

        calculate_metrics(log, (accum_loss/len(self.valid_dataloader)),
                          pred_list, gt_list, self.class2num_dict)
        
        """ Update best record """
        log["valid_state"] = ""
        if log[self.score_key] > self.best_val_f1:
            
            log["valid_state"] = "☆★☆ BEST_VALIDATION ☆★☆"
            self.best_val_f1 = log[self.score_key]
                        
            """ Update `best_val_log` """
            calculate_metrics(self.best_val_log, (accum_loss/len(self.valid_dataloader)),
                              pred_list, gt_list, self.class2num_dict)
            
            self.best_model_state_dict = deepcopy(self.model.state_dict())
            self.best_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
            self.best_val_log["epoch"] = epoch
            
            output_string += (f", ☆★☆ BEST_VALIDATION ☆★☆"
                              f", best_val_avg_loss = {self.best_val_log['average_loss']}"
                              f", best_val_{self.score_key} = {self.best_val_log[self.score_key]}")
        
        """ Check 'EarlyStop' """
        log["valid_improve"] = ""
        if self.enable_earlystop:
            if log["average_loss"] < self.best_val_avg_loss:
                self.best_val_avg_loss = log["average_loss"]
                self.accum_no_improved = 0
            else:
                log["valid_improve"] = "◎㊣◎ LOSS_NO_IMPROVED ◎㊣◎"
                self.accum_no_improved += 1
                output_string += (f", ◎㊣◎ LOSS_NO_IMPROVED ◎㊣◎"
                                  f", accum_no_improved = {self.accum_no_improved}")
        
        """ Update `self.valid_logs` """
        self.valid_logs.append(log)
        
        """ Update postfix of `pbar_n_valid` """
        temp_str = f" {'{'} Loss: {log['average_loss']}, "
        temp_str += f"{self.score_key}: {log[self.score_key]} {'}'} "
        self.pbar_n_valid.postfix = temp_str
        self.pbar_n_valid.refresh()
        
        """ Print `output_string` """
        if ("◎㊣◎" in output_string) or ("☆★☆" in output_string):
            self._cli_out.write(output_string)
        
        """ SystemExit condition """
        if self.accum_no_improved == self.max_no_improved:
            sys.exit() # raise SystemExit
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