import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from bayesmedaug.utils.model_utils import *
from bayesmedaug.models.unet import UNet
from bayesmedaug.utils.discrete import *
from bayesmedaug.augmentations.list import get_augmentations
from bayesmedaug.utils.loaders import get_dataloaders
from bayesmedaug.utils.model_utils import dice_loss
from bayesmedaug.models.eval import evaluate

def train_unet(epochs, batch_size, train_dir, eval_dir, initial_lr, scheduler, s_gamma, s_step, **params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)
    criterion = nn.CrossEntropyLoss()

    if scheduler:
        scheduler_fn = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = s_step,
            gamma = s_gamma
        )
 
    if "angle" in params.keys():
        params["angle"] = discrete_angle_normalized(params["angle"])

    if "shift_x" in params.keys():
        params["shift_x"] = discrete_shift(params["shift_x"])

    if "shift_y" in params.keys():
        params["shift_y"] = discrete_shift(params["shift_y"])

    transform = get_augmentations(**params)

    paths_ = []
    paths_.append(os.path.join(train_dir, "images"))
    paths_.append(os.path.join(train_dir, "labels"))
    paths_.append(os.path.join(eval_dir, "images"))
    paths_.append(os.path.join(eval_dir, "labels"))
    
    train_dataloader, test_dataloader = get_dataloaders(transform, paths_, batch_size)
    best_dice = 0
    total = len(train_dataloader) * epochs

    with tqdm(total = total, desc='Training round', leave=False, position=0, ) as tt:
        for epoch in range(epochs):
            batch_count, train_loss = 0, 0
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch_count +=1
                optimizer.zero_grad()
                out = model(batch["image"].to(device))

                loss = criterion(out, batch["mask"].long().to(device)) \
                    + dice_loss(
                        F.softmax(out, dim=1).float(),
                        F.one_hot(batch["mask"].to(device), model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                )

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                tt.update()

            val_score = evaluate(model, test_dataloader, device, True)
            if val_score.item() > best_dice:
                best_dice = val_score.item()

            if scheduler:
                scheduler_fn.step()

    return best_dice
