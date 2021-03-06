from typing import Optional
import bayesmedaug
from torchvision import transforms
import random
import warnings

#TODO:
# - optimize selection probability of each augmentation method
#   then match hyperparameters and methods automatically: Done

class Listed():
    def __init__(
            self,
            augmentations,
            optimize_p: bool = False,
            randomized: bool = True,
            randomized_bounds: tuple = (0.0001, 0.9999),
            prob_list: Optional[list] = None
    ):
        r"""
        Args:
            augmentations: the list of implemented augmentations that to be used.
            
            randomized: random applying probability for each augmentation.

            prob_list: the list of the probabilities if randomized is ``False``.
        """
        for a in augmentations:
            if a not in bayesmedaug.CURRENT_AUGMENTATIONS:
                raise NotImplementedError("No current implementation of this augmentation")
            
        self.augmentations = augmentations
        self.randomized = randomized
        self.randomized_bounds = randomized_bounds
        self.prob_list = prob_list
        self.optimize_p = optimize_p


    def __call__(self, **params):
        r_ = transforms.RandomApply
        f_ = transforms.Compose
        a_ = []
        used = []
        if self.randomized:
            for a in self.augmentations:
                p = random.uniform(self.randomized_bounds[0], self.randomized_bounds[1])
                hyperparams = a.hyperparameters()
                selected_hyperparams = {i: params[i] for i in hyperparams if i in params.keys()}
                used.extend(list(selected_hyperparams.keys()))
                a_.append(r_([a(**selected_hyperparams)], p = p))

        elif self.prob_list != None:
            for p, a in zip(self.prob_list, self.augmentations):
                hyperparams = a.hyperparameters()
                selected_hyperparams = {i: params[i] for i in hyperparams if i in params.keys()}
                used.extend(list(selected_hyperparams.keys()))
                a_append(r_([a(**selected_hyperparams)], p = p))

        elif self.optimize_p:
            for a in self.augmentations:
                hyperparams = a.hyperparameters()
                selected_hyperparams = {i: params[i] for i in hyperparams if i in params.keys()}
                used.extend(list(selected_hyperparams.keys()))
                a_.append(r_([a(**selected_hyperparams)], p = params["p_" + a.__name__]))
                used.append("p_" + a.__name__)

        composed = f_(a_)
        if set(used) != set(list(params.keys())):
            warnings.warn("WARNING: Listed data augmentations and hyperparameters are not matching.")
        return composed
