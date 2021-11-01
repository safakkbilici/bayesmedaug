# bayesmedaug: Bayesian Optimization Library for Medical Image Segmentation.

bayesmedaug optimizes your data augmentation hyperparameters for medical image segmentation tasks by using [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization).

bayesmedaug is currently in beta release and still in development.

## Authors
- M. Şafak Bilici
- Onur Boyar
- Enes Sadi Uysal
- Alara Hergün

## Simple Usage

```python
import torch
import bayesmedaug
from bayesmedaug import VanillaUNet, Trainer, Listed, BOMed
from bayesmedaug import Rotate, ZoomOut, Gamma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auglist = [
    Rotate,
    ZoomOut,
    Gamma
]

params = {
    'angle': (0,2),
    'zoom_amount': (0.5,0.9),
    'gamma': (0.5,1.5),
}

auglist = Listed(augmentations = auglist)

trainer = Trainer(
    model = VanillaUNet,
    model_args = {"n_channels": 1, "n_classes": 2},
    optimizer = torch.optim.Adam,
    optimizer_args = {"lr": 0.0005},
    device = device,
    epochs = 1,
    train_dir = "/home/safak/Desktop/drive/train/",
    eval_dir = "/home/safak/Desktop/drive/test/",
    augmentations = auglist,
    batch_size = 1
)


optimizer = BOMed(
    f = trainer.train,
    pbounds = params,
    random_state = 1,
)

optimizer.maximize(
    init_points = 15,
    n_iter = 15,
)
```
