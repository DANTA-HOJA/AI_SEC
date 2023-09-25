import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from collections import Counter
import json

import cv2
import numpy as np
from tqdm.auto import tqdm

import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, \
    GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

from ..imagetester.baseimagetester import BaseImageTester
from ..utils import rename_history_dir
from ...utils import calculate_metrics
from ....data.dataset.utils import parse_dataset_xlsx_name
from ....shared.utils import create_new_dir, formatter_padr0
# -----------------------------------------------------------------------------/


class BaseFishTester(BaseImageTester):


    def __init__(self) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__()
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        super()._set_attrs(config_file)
        
        """ Initial CAM generator and directory """
        if self.do_cam:
            self._set_cam_result_root()
            self._set_cam_generator()
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="3.2.test_by_fish.toml"):
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
        self._save_predict_ans_log() # save file
        self._save_test_log(test_desc="PredByFish", score_key="maweavg_f1") # save file
        self._save_report(test_desc="PredByFish") # save file
        
        """ Rename `history_dir` """
        # new_name_format : {time_stamp}_{test_desc}_{target_epochs_with_ImgLoadOptions}_{model_state}_{score}
        # example : '20230630_04_39_25_{Tested_PredByFish}_{100_epochs_AugOnFly}_{best}_{maweavg_f1_0.90208}'
        if self.do_cam:
            rename_history_dir(self.history_dir, "Tested_PredByFish_CAM",
                               self.model_state, self.test_log, score_key="maweavg_f1")
        else:
            rename_history_dir(self.history_dir, "Tested_PredByFish",
                               self.model_state, self.test_log, score_key="maweavg_f1")
        # ---------------------------------------------------------------------/



    def _set_config_attrs(self):
        """
        """
        super()._set_config_attrs()
        
        """ [cam] """
        self.do_cam: bool = self.config["cam"]["enable"]
        self.colormap_key: str = self.config["cam"]["colormap_key"]
        self.cam_colormap: int = getattr(cv2, self.colormap_key)
        # ---------------------------------------------------------------------/



    def _set_train_config_attrs(self):
        """
        """
        super()._set_train_config_attrs()
        
        self.crop_size: int = parse_dataset_xlsx_name(self.dataset_xlsx_name)["crop_size"]
        # ---------------------------------------------------------------------/



    def _set_cam_result_root(self):
        """
        """
        self.cam_result_root: Path = self.history_dir.joinpath("cam_result")
        assert not self.cam_result_root.exists(), \
            (f"Directory already exists: '{self.cam_result_root}'. "
             "To re-generate the cam results, "
             "please delete existing directory.")
        # ---------------------------------------------------------------------/



    def _set_cam_generator(self):
        """
        """
        self.cam_generator: GradCAM
        
        raise NotImplementedError("This is a base fish tester, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/



    def _set_testing_attrs(self):
        """
        """
        super()._set_testing_attrs()
        
        self.fish_pred_dict: Dict[str, Counter] = {}
        self.fish_gt_dict: Dict[str, str] = {}
        self.image_predict_ans_dict: dict = {}
        # ---------------------------------------------------------------------/



    def _one_epoch_testing(self):
        """
        """
        accum_loss: float = 0.0
        
        self.model.eval() # set to evaluation mode
        for batch, data in enumerate(self.test_dataloader):
            
            images, labels, crop_names = data
            images, labels = images.to(self.device), labels.to(self.device) # move to GPU
            
            preds = self.model(images)
            loss_value = self.loss_fn(preds, labels)
            accum_loss += loss_value.item() # accumulate current batch loss
                                            # tensor.item() -> get value of a Tensor

            """ Extend `pred_list`, `gt_list` """
            preds_prob = torch.nn.functional.softmax(preds, dim=1)
            _, preds_hcls = torch.max(preds_prob, 1) # get the highest probability class
            
            """ Generate CAM for current batch """
            if self.do_cam:
                """ If targets is `None`, returns the map for the highest scoring category.
                    Otherwise, targets the requested category.
                """
                grayscale_cam_batch = \
                    self.cam_generator(input_tensor=images, targets=None, 
                                       aug_smooth=True, eigen_smooth=True)
            
            """ Update 'predict_class' according to 'fish_dsname' """
            preds_hcls_list: list = preds_hcls.cpu().numpy().tolist()
            for crop_name, pred_hcls, label in zip(crop_names, preds_hcls_list, labels):
                self._update_fish_pred_gt_dict(crop_name, pred_hcls, label)
            
            """ Save cam result ( both of grayscale, color ) """
            if self.do_cam:
                for crop_name, grayscale_cam in zip(crop_names, grayscale_cam_batch):
                    self._save_cam_result(crop_name, grayscale_cam)

            """ Print number of matches in current batch """
            num_match = (preds_hcls.cpu() == labels.cpu()).sum().item()
            self._cli_out.write(f"Batch[ {(batch+1):{formatter_padr0(self.test_dataloader)}} / {len(self.test_dataloader)} ], "
                                f"# of (ground truth == prediction) in this batch : "
                                f"{num_match:{formatter_padr0(labels)}} / {len(labels)} "
                                f"( {num_match/len(labels):.2f} )")
            
            """ Update `pbar_n_test` """
            self.pbar_n_test.update(1)
            self.pbar_n_test.refresh()

        for key, value in self.fish_pred_dict.items(): self.fish_pred_dict[key] = value.most_common(1)[0][0]
        self.pred_list_to_name = [ value for _, value in self.fish_pred_dict.items() ]
        self.gt_list_to_name = [ value for _, value in self.fish_gt_dict.items() ]        
        
        calculate_metrics(self.test_log, (accum_loss/len(self.test_dataloader)),
                          self.pred_list_to_name, self.gt_list_to_name, self.class2num_dict)
        # ---------------------------------------------------------------------/



    def _update_fish_pred_gt_dict(self, crop_name:str,
                                        pred_hcls:int, label:int):
        """
        """
        crop_name_split: List[str] = crop_name.split("_")
        fish_dsname: str = "_".join(crop_name_split[:4]) # example_list : ['fish', '1', 'A', 'U', 'crop', '0']
                                                         # list[:3] = 'fish_1_A', list[:4] = 'fish_1_A_U'
        
        """ Update `self.fish_gt_dict` """
        fish_class: str = self.num2class_list[label]
        if fish_dsname not in self.fish_gt_dict:
            self.fish_gt_dict[fish_dsname] = fish_class
        
        """ Update `self.fish_pred_dict` """
        img_pred_class: str = self.num2class_list[pred_hcls]
        if fish_dsname not in self.fish_pred_dict:
            self.fish_pred_dict[fish_dsname] = Counter() # init a Counter
        self.fish_pred_dict[fish_dsname].update([img_pred_class])
        
        """ Store result for each crop image in `self.image_predict_ans_dict` ( for gallery ) """
        self.image_predict_ans_dict[crop_name] = { "gt": fish_class,
                                                   "pred": img_pred_class }
        # ---------------------------------------------------------------------/



    def _save_cam_result(self, crop_name:str, grayscale_cam):
        """
        """
        crop_name_split: List[str] = crop_name.split("_")
        fish_dsname: str = "_".join(crop_name_split[:4]) # example_list : ['fish', '1', 'A', 'U', 'crop', '0']
                                                         # list[:3] = 'fish_1_A', list[:4] = 'fish_1_A_U'
        resize: Tuple[int, int] = (self.crop_size, self.crop_size)
        
        """ Gray """
        cam_result_dir = self.cam_result_root.joinpath(fish_dsname, "grayscale_map")
        create_new_dir(cam_result_dir)
        crop_name_split[4] = "graymap"
        cam_save_path = cam_result_dir.joinpath(f"{'_'.join(crop_name_split)}.tiff")
        grayscale_cam = np.uint8(255 * grayscale_cam)
        cv2.imwrite(str(cam_save_path), \
                    cv2.resize(grayscale_cam, resize, interpolation=cv2.INTER_CUBIC))
        
        """ Color """
        cam_result_dir = self.cam_result_root.joinpath(fish_dsname, "color_map")
        create_new_dir(cam_result_dir)
        crop_name_split[4] = "colormap"
        cam_save_path = cam_result_dir.joinpath(f"{'_'.join(crop_name_split)}.tiff")
        colormap_cam = cv2.applyColorMap(grayscale_cam, self.cam_colormap) # BGR
        cv2.imwrite(str(cam_save_path), \
                    cv2.resize(colormap_cam, resize, interpolation=cv2.INTER_CUBIC))
        # ---------------------------------------------------------------------/



    def _save_predict_ans_log(self):
        """
        """
        file_name = r"{Logs}_PredByFish_predict_ans.log"
        save_path = self.history_dir.joinpath(file_name)
        with open(save_path, mode="w") as f_writer:
            json.dump(self.image_predict_ans_dict, f_writer, indent=4)
        # ---------------------------------------------------------------------/