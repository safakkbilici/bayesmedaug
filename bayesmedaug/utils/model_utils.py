import torch
from torch import Tensor
from sklearn import metrics


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    ### Calculates and returns the dice coefficient.
    #### Input Parameters ####
    #.....input -> TBA
    #.....target -> TBA
    #.....reduce_batch_first -> TBA
    #.....epsilon -> TBA
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Calculates the dice loss.
    #### Input Parameters ####
    #.....input -> TBA
    #.....target -> TBA
    #.....multiclass -> TBA
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def IoU(input: Tensor, target: Tensor):
  # Calculates the intersection over union value.
  #### Input Parameters ####
  #.....input -> TBA
  #.....target -> TBA

  assert input.size() == target.size()
  
  y_true_f = torch.flatten(target)
  y_pred_f = torch.flatten(input)

  intersection = torch.sum (y_true_f * y_pred_f)
  
  union = torch.sum (y_true_f + y_pred_f - y_true_f * y_pred_f)
  return intersection/union


def AUC(input: Tensor, target: Tensor):
  # Calculates the area under the curve value.
  #### Input Parameters ####
  #.....input -> TBA
  #.....target -> TBA

  assert input.size() == target.size()
  
  y_true_f = torch.flatten(target)
  y_pred_f = torch.flatten(input)

  y_true_f = Tensor.cpu(y_true_f)
  y_pred_f = Tensor.cpu(y_pred_f)

  fpr, tpr, thresholds = metrics.roc_curve(y_true_f, y_pred_f)
  auc_score = metrics.auc(fpr, tpr)

  return auc_score
