from bayesmedaug.models.unet import VanillaUNet
from bayesmedaug.callbacks.trainer import Trainer
from bayesmedaug.augmentations.alist import Listed
from bayesmedaug.callbacks.bo import BOMed
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
    RandomCrop,
    Emboss,
    Sharpen
]



def list_augmentations():
    for aug in CURRENT_AUGMENTATIONS:
        print(aug.__doc__)


def explain(obj_class):
    try:
        print(obj_class.__doc__)
    except:
        print("")
