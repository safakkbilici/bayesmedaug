import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Union, List
from tqdm.auto import tqdm

from bayesmedaug.augmentations.alist import Listed
from bayesmedaug.augmentations.binary import BinaryListed
from bayesmedaug.utils.loaders import (
    get_dataloaders,
    get_binary_dataloaders
)


from bayesmedaug.utils.model_utils import (
    dice_coeff,
    dice_loss,
    multiclass_dice_coeff,
    IoU,
    AUC
)

from bayesmedaug.utils.discretize import(
    roundup,
    discrete_angle,
    discrete_shift,
    discrete_rcrop,
    discrete_shear
)

class Trainer():
    def __init__(
            self,
            model,
            model_args: dict,
            optimizer: torch.optim,
            optimizer_args: dict,
            device: torch.device,
            epochs: int,
            train_dir: str,
            eval_dir: str,
            augmentations: Union[Listed, BinaryListed],
            scheduler: Optional = None,
            scheduler_args: Optional[dict]  = None,
            batch_size: int = 5,
            return_best: bool = True,
            dice_loss: bool = True,
            return_metric : str = 'dice',
            print_metrics: Optional[List[str]] = None
    ) -> None:
        r"""
        Args:
            model: implemented unet models that to be used, i.e., bayesmedaug.VanillaUNet
         
            model_args: the dictionary for model's attributes.

            optimizer: pytorch optimizer, i.e., torch.optim.Adam

            optimizer_args: the dictionary for optimizer's attributes.

            device: hardware device.

            epochs: number of epochs for each iteration of bayesian optimization.

            train_dir: train image path, i.e., ``Desktop/drive/train/``

            eval_dir: eval image path, i.e., ``Desktop/drive/test/``

            augmentations: the object of the Listed class.

            scheduler: learning rate scheduler if desired, i.e., torch.optim.lr_Scheduler.StepLR

            scheduler_args: the dictionary for scheduler's attributes.

            batch_size: batch size for both training and evaluating.

            return_best: bayesian optimization selects best mean dice score over all epoch results.

            dice_loss: if True, then the loss function becomes crossentropy + dice, else crossentropy

            return_metric: Bayesian Optimization optimizes with given metric. Default is 'dice', can also be 'iou' or 'auc'.
        """
        self.model = model
        self.model_args = model_args
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args
        self.device = device
        self.return_best = return_best
        self.train_dir = train_dir
        self.eval_dir = eval_dir
        self.dice_loss = dice_loss
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.epochs = epochs
        self.return_metric = return_metric
        self.print_metrics = print_metrics
        self.implemented_metrics = ["iou", "auc", "dice"]

    def train(self, **params):
        model = self.model(**self.model_args).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = self.optimizer(model.parameters(), **self.optimizer_args)
        if self.scheduler != None:
            scheduler = self.scheduler(optimizer, **self.scheduler_args)
        
        if "angle" in params.keys():
            params["angle"] = discrete_angle(params["angle"])
        if "shift_amount_x" in params.keys():
            params["shift_amount_x"] = discrete_shift(params["shift_amount_x"])
        if "shift_amount_y" in params.keys():
            params["shift_amount_y"] = discrete_shift(params["shift_amount_y"])
        if "crop_height" in params.keys():
            params["crop_height"] = discrete_rcrop(params["crop_height"])
        if "shear_amount_y" in params.keys():
            params["shear_amount_y"] = discrete_shear(params["shear_amount_y"])
        if "shear_amount_x" in params.keys():
            params["shear_amount_x"] = discrete_shear(params["shear_amount_x"])


        if type(self.augmentations) == Listed:
            transform = self.augmentations(**params)
        elif type(self.augmentations) == BinaryListed:
            self.augmentations(**params)

        paths_ = []
        paths_.append(os.path.join(self.train_dir, "images"))
        paths_.append(os.path.join(self.train_dir, "labels"))
        paths_.append(os.path.join(self.eval_dir, "images"))
        paths_.append(os.path.join(self.eval_dir, "labels"))

        if type(self.augmentations) == Listed:
            train_dataloader, test_dataloader = get_dataloaders(transform, paths_, self.batch_size)
        elif type(self.augmentations) == BinaryListed:
            train_dataloader, test_dataloader = get_binary_dataloaders(paths_, self.batch_size)
            
        total = len(train_dataloader) * self.epochs
        
        best_metric = 0

        with tqdm(total=total, desc="Training Round", leave=False, position=0) as tt:
            for epoch in range(self.epochs):
                batch_count, train_loss = 0, 0
                for step, batch in enumerate(train_dataloader):
                    model.train()
                    batch_count +=1
                    optimizer.zero_grad()
                    if "image_aug" in batch.keys() and batch["image_aug"].flatten().sum() != 0:
                        images = torch.cat([batch["image"], batch["image_aug"]], dim=0)
                        masks = torch.cat([batch["mask"], batch["mask_aug"]], dim=0)
                    else:
                        images = batch["image"]
                        masks = batch["mask"]
                        
                    out = model(images.to(self.device))

                    if self.dice_loss:
                        loss = criterion(out, masks.long().to(self.device)) \
                            + dice_loss(
                                F.softmax(out, dim=1).float(),
                                F.one_hot(masks.to(self.device), model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    else:
                        loss = criterion(out, masks.long().to(self.device))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    tt.update()

                val_score = self.evaluate(model, test_dataloader, self.device, True)

                if self.return_best:
                    if val_score.item() > best_metric:
                        try:
                          best_metric = val_score.item()
                        except:
                          best_metric = float(val_score)
                else:
                    try:
                      best_metric = val_score.item()
                    except:
                      best_metric = float(val_score)

                if self.scheduler != None:
                    scheduler.step()

        if self.print_metrics != None:
            pm = self.pmetric(model, test_dataloader, self.device, True)
            print(pm)
        return best_metric

    @torch.no_grad()
    def evaluate(self, net, dataloader, device, disable=True):
        net.eval()
        num_val_batches = len(dataloader)
        eval_score = 0

        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation Round', unit='batch', leave=False, position=0, disable=disable):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

            mask_pred = net(image)
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                if self.return_metric == 'dice':
                    eval_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                elif self.return_metric == 'iou':
                    eval_score += IoU(mask_pred, mask_true)
                elif self.return_metric == 'auc':
                    eval_score += AUC(mask_pred, mask_true)          
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                if self.return_metric == 'auc':
                    eval_score += AUC(mask_pred, mask_true)
                elif self.return_metric == 'dice':
                    eval_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                elif self.return_metric == 'iou':
                    eval_score += IoU(mask_pred, mask_true)
            
        return eval_score / num_val_batches

    @torch.no_grad()
    def pmetric(self, net, dataloader, device, disable = True, round_ = 3):
        metrics = {k: 0 for k in self.print_metrics}
        net.eval()
        num_val_batches = len(dataloader)
        auc, dice, iou = 0, 0, 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation Round', unit='batch', leave=False, position=0, disable=disable):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

            mask_pred = net(image)
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                if "dice" in self.print_metrics:
                    metrics["dice"] += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                elif "iou" in self.print_metrics:
                    metrics["iou"] += IoU(mask_pred, mask_true)
                elif "auc" in self.print_metrics:
                    metrics["auc"] += AUC(mask_pred, mask_true)          
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                if "auc" in self.print_metrics:
                    metrics["auc"] += AUC(mask_pred, mask_true)
                elif "dice" in self.print_metrics:
                    metrics["dice"] += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                elif "iou" in self.print_metrics:
                    metrics["iou"] += IoU(mask_pred, mask_true)

        print_metrics = {k: round(v/num_val_batches, round_) for k, v in metrics.items()}
        return print_metrics
