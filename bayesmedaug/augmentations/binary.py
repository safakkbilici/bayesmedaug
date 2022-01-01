import cv2
import os, glob, shutil
import bayesmedaug

from typing import List

dtypes = [
    "jpg",
    "png",
    "jpeg"
]

class BinaryGenerator():
    def __init__(
            self,
            augmentations: List,
            train_image_path: str,
            train_mask_path: str,
            tmp_path: str
    ) -> None:
        r"""
        Args:
            augmentations: the list of current implemented augmentations

            train_image_path: train image directory

            train_mask_path: train mask directory

            tmp_image_path: temporary image path with augmentations

            tmp_mask_path: temporary mask path with augmentations
        """                      

        self.augmentations = augmentations
        mask = []
        image = []
        
        if train_image_path[-1] != "/":
            self.train_image_path = train_image_path + "/"

        if train_mask_path[-1] != "/":
            self.train_mask_path = train_mask_path + "/"

        if tmp_path[-1] != "/":
            self.tmp_path = tmp_path + "/"

        self.tmp_image_path = self.tmp_path + "images/"
        self.tmp_mask_path = self.tmp_path + "labels/"

        if os.path.exists(self.tmp_image_path):
            shutil.rmtree(self.tmp_image_path)

        if os.path.exists(self.tmp_mask_path):
            shutil.rmtree(self.tmp_mask_path)

        os.makedirs(self.tmp_image_path)
        os.makedirs(self.tmp_mask_path)
            

        for dtype in dtypes:
            select_i = glob.glob(f"{train_image_path}"+f"*.{dtype}")
            select_m = glob.glob(f"{train_mask_path}"+f"*.{dtype}")
            if select_i and select_m:
                mask.extend(select_m)
                image.extend(select_i)

        self.mask = mask
        self.image = image


    def __call__(self):
        for image_str, mask_str in zip(self.image, self.mask):
            
            img = cv2.imread(image_str, 0)
            mask = cv2.imread(mask_str, 0)
            
            img_prefix = ''.join(image_str.split(".")[:-1]) + "_"
            mask_prefix = ''.join(mask_str.split(".")[:-1]) + "_"
            
            img_name = os.path.basename(image_str).split(".")[0]
            mask_name = os.path.basename(mask_str).split(".")[0]
            
            img_extension = image_str.split(".")[-1]
            mask_extension = mask_str.split(".")[-1]

            cv2.imwrite(
                self.tmp_image_path + img_name + "." + img_extension,
                img
            )

            cv2.imwrite(
                self.tmp_mask_path + mask_name + "." + mask_extension,
                mask
            )

            for aug in self.augmentations:
                if type(aug) != list:
                    img_aug, mask_aug = aug((img, mask))
                    cv2.imwrite(
                        self.tmp_image_path + img_name + "_" +aug.__class__.__name__ + "." + img_extension,
                        img_aug
                    )

                    cv2.imwrite(
                        self.tmp_mask_path + mask_name + "_" + aug.__class__.__name__ + "." + mask_extension,
                        mask_aug
                    )
                    

                else:
                    img_aug = np.copy(img)
                    mask_aug= np.copy(mask)
                    aug_prefix = ""
                    for a in augs:
                        img_aug, mask_aug = a((img_aug, mask_aug))
                        aug_prefix += a.__class__.__name__

                    cv2.imwrite(
                        self.tmp_image_path + img_name + "_" + aug_prefix + "." + img_extension,
                        img_aug
                    )

                    cv2.imwrite(
                        self.tmp_mask_path + mask_name + "_" + aug_prefix + "." + mask_extension,
                        mask_aug
                    )        

class BinaryListed():
    def __init__(
            self,
            augmentations,
            train_image_path,
            train_mask_path,
            tmp_path
    ):
        for a in augmentations:
            if type(a) != list:
                if a not in bayesmedaug.CURRENT_AUGMENTATIONS:
                    raise NotImplementedError("No current implementation of this augmentation")
            elif type(a) == List:
                for a_ in a:
                    if a_ not in bayesmedaug.CURRENT_AUGMENTATIONS:
                        raise NotImplementedError("No current implementation of this augmentation")
            else:
                raise NotImplementedError("No current implementation of this augmentation")
        
        self.augmentations = augmentations
        self.train_image_path = train_image_path
        self.train_mask_path = train_mask_path
        self.tmp_path = tmp_path


    def __call__(self, **params):
        do = []
        for aug in self.augmentations:
            if type(aug) != list:
                if params["p_"+aug.__name__] >= 0.5:
                    hyperparams = aug.hyperparameters()
                    selected_hyperparams = {i: params[i] for i in hyperparams if i in params.keys()}
                    
                    obj_a = aug(**selected_hyperparams)
                    do.append(obj_a)
            elif type(aug) == list:
                aug_p_name = "p_"
                for a in aug:
                    aug_p_name = aug_p_name + a.__name__
                if params["p_"+aug_p_name] >= 0.5:
                    do.append(aug)

        data_generator = BinaryGenerator(
            augmentations = do,
            train_image_path = self.train_image_path,
            train_mask_path = self.train_mask_path,
            tmp_path = self.tmp_path
        )

        data_generator()
                
