# bayesmedaug: Bayesian Optimization Library for Medical Image Segmentation.

bayesmedaug optimizes your data augmentation hyperparameters for medical image segmentation tasks by using [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization).

bayesmedaug is currently in beta release and still in development.

## Optimizing Magnitudes

```python
import torch
import bayesmedaug
from bayesmedaug import VanillaUNet, Trainer, Listed, BOMed
from bayesmedaug import Rotate, ZoomOut, Gamma, RandomCrop, Sharpen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auglist = [
    Rotate,
    ZoomOut,
    Gamma,
    RandomCrop,
    Sharpen
]

params = {
    'angle': (0.2,2.8),
    'zoom_amount': (0.5,0.9),
    'gamma': (0.5,1.5),
    'crop_height': (0.1, 0.4),
    'alpha': (0.1, 1),
    'lightness': (0.1, 1.5),

}

auglist = Listed(augmentations = auglist)

trainer = Trainer(
    model = VanillaUNet,
    model_args = {"n_channels": 1, "n_classes": 2},
    optimizer = torch.optim.Adam,
    optimizer_args = {"lr": 0.0005},
    device = device,
    epochs = 1,
    train_dir = "~/Desktop/drive/train/",
    eval_dir = "~/Desktop/drive/test/",
    augmentations = auglist,
    batch_size = 1
)
```

## Optimizing Probabilities and Magnitudes

```python
import torch
import bayesmedaug
from bayesmedaug import VanillaUNet, Trainer, Listed, BOMed
from bayesmedaug import Rotate, ZoomOut, Gamma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auglist = [
    Rotate,
    ZoomOut,
    Gamma,
    RandomCrop,
    Sharpen
]

params = {
    'angle': (0.2,2.8),                                                         
    'zoom_amount': (0.5,0.9),                                                   
    'gamma': (0.5,1.5),                                                         
    'crop_height': (0.1, 0.4),                                                  
    'alpha': (0.1, 1),                                                          
    'lightness': (0.1, 1.5),                                                    
    'p_'+Rotate.__name__: (0, 0.7),                                             
    'p_'+ZoomOut.__name__: (0, 0.7),                                            
    'p_'+Gamma.__name__: (0, 0.7),                                              
    'p_'+RandomCrop.__name__: (0, 0.7),                                         
    'p_'+Sharpen.__name__: (0, 0.7)
}

auglist = Listed(augmentations = auglist, optimize_p = True, randomized = False)

trainer = Trainer(
    model = VanillaUNet,
    model_args = {"n_channels": 1, "n_classes": 2},
    optimizer = torch.optim.Adam,
    optimizer_args = {"lr": 0.0005},
    device = device,
    epochs = 1,
    train_dir = "~/Desktop/drive/train/",
    eval_dir = "~/Desktop/drive/test/",
    augmentations = auglist,
    batch_size = 1
)
```

## Optimizing Binary Probabilities and Magnitudes

```python
import torch
import bayesmedaug
from bayesmedaug import VanillaUNet, Trainer, BinaryListed, BOMed
from bayesmedaug import Rotate, ZoomOut, Gamma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tmp_path = "./tmp"

auglist = [
    Rotate,
    ZoomOut,
    Gamma,
    RandomCrop,
    Sharpen
]

params = {
    'angle': (0.2,2.8),                                                         
    'zoom_amount': (0.5,0.9),                                                   
    'gamma': (0.5,1.5),                                                         
    'crop_height': (0.1, 0.4),                                                  
    'alpha': (0.1, 1),                                                          
    'lightness': (0.1, 1.5),                                                    
    'p_'+Rotate.__name__: (0, 1),                                             
    'p_'+ZoomOut.__name__: (0, 1),                                            
    'p_'+Gamma.__name__: (0, 1),                                              
    'p_'+RandomCrop.__name__: (0, 1),                                         
    'p_'+Sharpen.__name__: (0, 1)
}

auglist = BinaryListed(
    augmentations = auglist,
    train_image_path = "~/Desktop/drive/train/images",
    train_mask_path = "~/Desktop/drive/train/labels",
    tmp_path = "./tmp",
)

trainer = Trainer(
    model = VanillaUNet,
    model_args = {"n_channels": 1, "n_classes": 2},
    optimizer = torch.optim.Adam,
    optimizer_args = {"lr": 0.0005},
    device = device,
    epochs = 1,
    train_dir = tmp_path,
    eval_dir = "~/Desktop/drive/test/",
    augmentations = auglist,
    batch_size = 1
)
```

## Then:

```python
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


## Authors
- [M. Şafak Bilici](https://github.com/safakkbilici)
- [Onur Boyar](https://github.com/onurboyar)
- [Enes Sadi Uysal](https://github.com/eneSadi)
- [Alara Hergün](https://github.com/alarahergun)
