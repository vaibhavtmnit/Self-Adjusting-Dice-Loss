import torch
import xgboost as xgb
import numpy as np


class SadlTorch(torch.nn.Module):

    """
    Calcuates loss based on "Dice Loss for Data-imbalanced NLP Tasks" paper.
    Highlights of the loss:
    - Even negative misclassification adds to the final loss due to gamma(default=1) term
    - due to ((1-pij)**alpha)*pij term model puts less emphasis on easy examples and focus more on hard examples. Without this term its difficult for model to distinguish hard
    negative and positive ones

    alpha (float): factor to reduce loss contribution of easy examples
    gamma (float): smoothing factor to contribute to loss term for mis classified negative terms
    reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    sq_den (bool): Flag to get denominator squared. Paper suggests squaring will lead to faster convergence (Ref. Fully convolutional neural networks for volumetric medical image segmentation. In
2016 Fourth International Conference on 3D Vision(3DV), pages 565–571. IEEE.)
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean", sq_den=True) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        decayed_probs = ((1 - probs) ** self.alpha) * probs

        if sq_den:

            loss = 1 - (2 * decayed_probs  + self.gamma) / (decayed_probs  + 1 + self.gamma)

        else:
            loss = 1 - (2 * decayed_probs + self.gamma) / (torch.square(decayed_probs + 1 + self.gamma) 


        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
    


class SadlGbt():

    """
    Calcuates loss based on "Dice Loss for Data-imbalanced NLP Tasks" paper. This class is compatible with tree based models like XGBoost, LightGBM etc. This is compatible with 
    native python API not the sklearn wrapper.
    Highlights of the loss:
    - Even negative misclassification adds to the final loss due to gamma(default=1) term
    - due to ((1-pij)**alpha)*pij term model puts less emphasis on easy examples and focus more on hard examples. Without this term its difficult for model to distinguish hard
    negative and positive ones

    alpha (float): factor to reduce loss contribution of easy examples
    gamma (float): smoothing factor to contribute to loss term for mis classified negative terms 
    
    """


    def __init__(self, alpha: float = 1.0, gamma: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        
    def sfx(self, x):

        x = x - x.max(axis = 0, keepdims=True)
        y = np.exp(x)

        return y / y.sum(axis = 0, keepdims=True)

    def power_trns(self, x, pow_):

        return np.sign(x) * (np.abs(x)) ** pow_


    def sadl(self, logits: np.array, targets: xgb.dmatrix) :

        label = targets.get_label()

        # get prediction
        probs = self.sfx(logits)
        probs = np.take_along_axis(probs, label)

        g1 = probs * (1 - probs)
        g2 = label + ((-1) ** label) * probs
        g3 = probs + label - 1
        g4 = 1 - label - ((-1) ** label) * probs
        g5 = label + ((-1) ** label) * probs
        # combine the gradient
        grad = self.gamma * g3 * self.power_trns(g2, gamma_indct) * np.log(g4 + 1e-9) + \
               ((-1) ** label) * self.power_trns(g5, (gamma_indct + 1))
        # combine the gradient parts to get hessian components
        hess_1 = self.power_trns(g2, gamma_indct) + \
                 gamma_indct * ((-1) ** label) * g3 * self.power_trns(g2, (gamma_indct - 1))
        hess_2 = ((-1) ** label) * g3 * self.power_trns(g2, gamma_indct) / g4
        # get the final 2nd order derivative
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct +
                (gamma_indct + 1) * self.power_trns(g5, gamma_indct)) * g1

        return grad, hess


        