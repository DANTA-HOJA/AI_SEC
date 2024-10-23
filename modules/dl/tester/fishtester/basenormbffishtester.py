import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from pytorch_grad_cam import (AblationCAM, DeepFeatureFactorization, EigenCAM,
                              FullGrad, GradCAM, GradCAMPlusPlus, HiResCAM,
                              ScoreCAM, XGradCAM)
from tqdm.auto import tqdm

from ....shared.utils import (create_new_dir, formatter_padr0,
                              get_target_str_idx_in_list)
from ...utils import calculate_metrics
from ..imagetester.basenormbfimagetester import BaseNormBFImageTester
from ..utils import rename_history_dir
# -----------------------------------------------------------------------------/


class BaseNormBFFishTester(BaseNormBFImageTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI=display_on_CLI)
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]): # extend
        """
        """
        super()._set_attrs(config)
        
        """ Initial CAM generator and directory """
        if self.do_cam:
            self._cli_out.write(f"※　: Do CAM, colormap using '{self.colormap}'")
            self._set_cam_result_root()
            self._set_cam_generator() # abstract function
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self): # extend
        """
        """
        super()._set_config_attrs()
        
        """ [cam] """
        self.do_cam: bool = self.config["cam"]["enable"]
        self.colormap: str = self.config["cam"]["colormap"]
        # ---------------------------------------------------------------------/


    def _set_cam_result_root(self):
        """
        """
        self.cam_result_root: Path = self.history_dir.joinpath("cam_result")
        if self.cam_result_root.exists():
            raise FileExistsError(
                f"Directory already exists: '{self.cam_result_root}'. "
                "To re-generate the cam results, "
                "please delete existing directory.")
        # ---------------------------------------------------------------------/


    def _set_cam_generator(self): # abstract function
        """
        """
        self.cam_generator: GradCAM
        
        raise NotImplementedError("This is a base fish tester, \
            you should create a child class and replace this funtion")
        # ---------------------------------------------------------------------/


    # def _set_dff(self): # abstract function
    #     """
    #     """
    #     self.dff: DeepFeatureFactorization
        
    #     raise NotImplementedError("This is a base fish tester, \
    #         you should create a child class and replace this funtion")
    #     # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super(BaseNormBFImageTester, self).run(config)
        
        self._save_testing_amount_file() # save file
        
        """ Testing """
        self._set_testing_attrs()
        self._cli_out.divide()
        self.pbar_n_test = tqdm(total=len(self.test_dataloader),
                                desc="Test (PredByFish) ")
        
        self._one_epoch_testing()
        
        self.pbar_n_test.close()
        self._cli_out.new_line()
        
        """ Save files """
        self._save_predict_ans_log() # save file
        self._save_test_log(test_desc="PredByFish", score_key="maweavg_f1") # save file
        self._save_report(test_desc="PredByFish") # save file
        self._save_confusion_matrix_display(test_desc="PredByFish") # save file
        
        """ Rename `history_dir` """
        # new_name_format : {time_stamp}_{test_desc}_{target_epochs_with_ImgLoadOptions}_{model_state}_{score}
        # example : '20230630_04_39_25_{Tested_PredByFish}_{100_epochs_AugOnFly}_{best}_{maweavg_f1_0.90208}'
        if (self.do_cam) or (self.history_dir.joinpath("cam_result").exists()):
            rename_history_dir(self.history_dir, "Tested_PredByFish_CAM",
                               self.model_state, self.test_log, score_key="maweavg_f1",
                               cli_out=self._cli_out)
        else:
            rename_history_dir(self.history_dir, "Tested_PredByFish",
                               self.model_state, self.test_log, score_key="maweavg_f1",
                               cli_out=self._cli_out)
        
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _set_testing_attrs(self): # extend
        """
        """
        super()._set_testing_attrs()
        
        self.fish_pred_dict: Dict[str, Counter] = {}
        self.fish_gt_dict: Dict[str, Counter] = {}
        self.image_predict_ans_dict: dict = {}
        # ---------------------------------------------------------------------/


    def _one_epoch_testing(self): # overwrite
        """
        """
        accum_loss: float = 0.0
        
        self.model.eval() # set to evaluation mode
        for batch, data in enumerate(self.test_dataloader):
            
            images, _, labels, crop_names = data
            images, labels = images.to(self.device), labels.to(self.device) # move to GPU
            
            preds = self.model(images)
            loss_value = self.loss_fn(preds, labels)
            
            """ Accumulate current batch loss """
            accum_loss += loss_value.item() # tensor.item() -> get value of a Tensor

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
            
            # # # Deep Feature Factorizations
            # from pytorch_grad_cam.utils.image import show_factorization_on_image
            # from copy import deepcopy
            # from PIL import Image
            # self._set_dff()
            # tmp_df = deepcopy(self.test_df).set_index("image_name")
            # for image, name, pred, label in zip(images, crop_names, preds_hcls, labels):
            #     print(f"name, pred, label = {name, pred, label}")
            #     image = image.unsqueeze(0).cpu()
            #     self.model.to("cpu")
            #     concepts, batch_explanations, concept_outputs = self.dff(image, 3)
            #     concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()
            #     concept_label_strings = self.create_labels(concept_outputs, top_k=1)
                
            #     img = np.array(Image.open(self.src_root.joinpath(tmp_df.loc[name, "path"])).resize((224, 224)))
            #     rgb_img_float = np.float32(img) / 255
            #     visualization = show_factorization_on_image(rgb_img_float, 
            #                                                 batch_explanations[0],
            #                                                 image_weight=0.3,
            #                                                 concept_labels=concept_label_strings)
            #     result = np.hstack((img, visualization))
            #     Image.fromarray(result).save(self.history_dir.joinpath("dff.png"))
            # self.model.to(self.device)
            
            """ Update 'predict_class' according to 'fish_dsname' """
            preds_hcls_list: list = preds_hcls.cpu().numpy().tolist()
            preds_prob_array: np.ndarray = preds_prob.cpu().detach().numpy()
            for crop_name, pred_prob, pred_hcls, label in \
                    zip(crop_names, preds_prob_array, preds_hcls_list, labels):
                self._update_fish_pred_gt_dict(crop_name, pred_prob, pred_hcls, label)
            
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

        for key, cnt in self.fish_pred_dict.items(): self.fish_pred_dict[key] = cnt.most_common(1)[0][0]
        for key, cnt in self.fish_gt_dict.items(): self.fish_gt_dict[key] = cnt.most_common(1)[0][0]
        self.pred_list_to_name = [ value for _, value in self.fish_pred_dict.items() ]
        self.gt_list_to_name = [ value for _, value in self.fish_gt_dict.items() ]        
        
        calculate_metrics(self.test_log, (accum_loss/len(self.test_dataloader)),
                          self.pred_list_to_name, self.gt_list_to_name, self.class2num_dict)
        # ---------------------------------------------------------------------/


    def _update_fish_pred_gt_dict(self, crop_name:str, pred_prob: np.ndarray,
                                        pred_hcls:int, label:int):
        """
        """
        crop_name_split: List[str] = crop_name.split("_")
        target_idx = get_target_str_idx_in_list(crop_name_split, "crop")
        fish_dsname: str = "_".join(crop_name_split[:target_idx])
        # Example : ['fish', '1', 'A', 'U', 'crop', '0']
        # >>> list[:3] = 'fish_1_A'; list[:4] = 'fish_1_A_U'
        
        """ Update `self.fish_gt_dict` """
        gt_class: str = self.num2class_list[label]
        if fish_dsname not in self.fish_gt_dict:
            self.fish_gt_dict[fish_dsname] = Counter() # init a Counter
        self.fish_gt_dict[fish_dsname].update([gt_class])
        
        """ Update `self.fish_pred_dict` """
        pred_class: str = self.num2class_list[pred_hcls]
        if fish_dsname not in self.fish_pred_dict:
            self.fish_pred_dict[fish_dsname] = Counter() # init a Counter
        self.fish_pred_dict[fish_dsname].update([pred_class])
        
        """ Store result for each crop image in `self.image_predict_ans_dict` ( for gallery ) """
        tmp_dict = {}
        for k, v in self.class2num_dict.items():
            tmp_dict[k] = round(float(pred_prob[v]), 5)
        self.image_predict_ans_dict[crop_name] = { "gt": gt_class,
                                                   "pred": pred_class,
                                                   "pred_prob": tmp_dict}
        # ---------------------------------------------------------------------/


    def _save_cam_result(self, crop_name:str, grayscale_cam):
        """
        """
        crop_name_split: List[str] = crop_name.split("_")
        target_idx = get_target_str_idx_in_list(crop_name_split, "crop")
        fish_dsname: str = "_".join(crop_name_split[:target_idx])
        # Example : ['fish', '1', 'A', 'U', 'crop', '0']
        # >>> list[:3] = 'fish_1_A'; list[:4] = 'fish_1_A_U'
        
        resize: Tuple[int, int] = \
            (self.test_set.crop_size, self.test_set.crop_size)
        
        """ Gray """
        cam_result_dir = self.cam_result_root.joinpath(fish_dsname, "grayscale_map")
        create_new_dir(cam_result_dir)
        crop_name_split[target_idx] = "graymap"
        cam_save_path = cam_result_dir.joinpath(f"{'_'.join(crop_name_split)}.tiff")
        grayscale_cam = np.uint8(255 * grayscale_cam)
        cv2.imwrite(str(cam_save_path), \
                    cv2.resize(grayscale_cam, resize, interpolation=cv2.INTER_CUBIC))
        
        """ Color """
        cam_result_dir = self.cam_result_root.joinpath(fish_dsname, "color_map")
        create_new_dir(cam_result_dir)
        crop_name_split[target_idx] = "colormap"
        cam_save_path = cam_result_dir.joinpath(f"{'_'.join(crop_name_split)}.tiff")
        color_cam = cv2.applyColorMap(grayscale_cam,
                                        getattr(cv2, self.colormap)) # BGR
        cv2.imwrite(str(cam_save_path), \
                    cv2.resize(color_cam, resize, interpolation=cv2.INTER_CUBIC))
        # ---------------------------------------------------------------------/


    def _save_predict_ans_log(self):
        """
        """
        file_name = r"{Logs}_PredByFish_predict_ans.log"
        save_path = self.history_dir.joinpath(file_name)
        with open(save_path, mode="w") as f_writer:
            json.dump(self.image_predict_ans_dict, f_writer, indent=4)
        # ---------------------------------------------------------------------/


    # def create_labels(self, concept_scores, top_k=2):
    #     """ Deep Feature Factorizations
        
    #         Create a list with the image-net category names of the top scoring categories
    #     """
    #     concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    #     concept_labels_topk = []
    #     for concept_index in range(concept_categories.shape[0]):
    #         categories = concept_categories[concept_index, :]
    #         concept_labels = []
    #         for category in categories:
    #             score = concept_scores[concept_index, category]
    #             label = f"{self.num2class_list[category]}: {score:.2f}"
    #             concept_labels.append(label)
    #         concept_labels_topk.append("\n".join(concept_labels))
    #     return concept_labels_topk
    #     # ---------------------------------------------------------------------/