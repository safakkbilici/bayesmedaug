from bayesmedaug.models.unet import VanillaUNet
from bayesmedaug.callbacks.trainer import Trainer
from bayesmedaug.augmentations.functional import *



CURRENT_AUGMENTATIONS = [
    Gamma,
    GaussianBlur,
    OpticalDistortion,
    GaussianNoise,
    GaussianNoiseDeterministic,
    BrightnessContrast,
    Rotate,
    ShearX,
    ShearY,
    ShiftX,
    ShiftY,
    ZoomOut,
    RandomCrop
]



def list_augmentations():
    for aug in CURRENT_AUGMENTATIONS:
        print(aug.__doc__)
