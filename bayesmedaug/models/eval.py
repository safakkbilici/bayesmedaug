import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesmedaug.utils.model_utils import *

@torch.no_grad()
def evaluate(net, dataloader, device, disable=True):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False, position=0, disable=disable):
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        #with torch.no_grad():
        mask_pred = net(image)
        if net.n_classes == 1:
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        else:
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            
    return dice_score / num_val_batches
