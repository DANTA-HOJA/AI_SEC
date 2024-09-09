from imgaug import augmenters as iaa
# -----------------------------------------------------------------------------/


def crop_base_size(width, height):
    """
    """
    return iaa.CropToFixedSize(width=width, height=height,
                               position="center")
    # -------------------------------------------------------------------------/



def dynamic_crop(size:int) -> iaa.Sequential:
    """
    random crop function: https://github.com/aleju/imgaug/issues/31
    
    action: rotate \
            -> random crop \
                -> others (apply before `too dark` detection, for ablation) \
                    -> return
    """
    transform = iaa.Sequential([
        iaa.Sometimes(0.5, 
            iaa.Affine(rotate=(-90, 90)),
        ),
        iaa.CropToFixedSize(width=size, height=size),
        # iaa.GammaContrast((0.5, 2.0)),
        # iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
    ])
    
    return transform
    # -------------------------------------------------------------------------/



def composite_aug() -> iaa.Sequential:
    """
    """
    transform = iaa.Sequential([
        # iaa.Sometimes(0.5, iaa.Affine(
        #     # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     rotate=(-25, 25),
        #     shear=(-8, 8)
        # )),
        # iaa.CropToFixedSize(width=512, height=512),
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        iaa.Sequential([
            # iaa.Sometimes(0.1, iaa.WithChannels([1], iaa.Clouds())), # ch_G
            # iaa.Sometimes(0.1, iaa.WithChannels([0, 1], iaa.Clouds())), # ch_B, ch_G
            # iaa.Sometimes(0.3, iaa.Cartoon()),
            iaa.Sometimes(0.3, [
                iaa.OneOf([
                    iaa.GammaContrast((0.5, 2.0)), # 可能會調得更暗，暫時取消
                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                ])
            ]),
            iaa.Sometimes(0.3, [
                iaa.OneOf([
                    iaa.GaussianBlur(), # default sigma is (0, 3)
                    iaa.Sharpen(),
                ])
            ]),
        ], random_order=True),
        # iaa.Dropout2d(p=0.2, nb_keep_channels=2),
    ])
    
    return transform
    # -------------------------------------------------------------------------/



# _cloud = iaa.CloudLayer(intensity_mean=[100,170],
#                         intensity_freq_exponent=[-2.5, -1.5],
#                         intensity_coarse_scale=[0,5],
#                         alpha_min=0.0,
#                         alpha_multiplier=[0.3, 1.0],
#                         alpha_freq_exponent=[-4.0, -1.5],
#                         sparsity=1.0,
#                         density_multiplier=[0.5, 1.5],
#                         alpha_size_px_max=3)



def fake_autofluorescence():
    """
    """
    aug = iaa.Sequential([
        iaa.Sometimes(0.1, iaa.WithChannels([0, 1], iaa.Clouds())), # ch_B, ch_G
        # iaa.Dropout2d(p=0.2, nb_keep_channels=2),
    ])
    
    return aug
    # -------------------------------------------------------------------------/