import torch
import torch.nn.functional as F

import attacks
import config
from attacks.attacks import AttackInstance


class NoAttack(AttackInstance):
    """
    Implementation of the typical PGD attack, as described in https://arxiv.org/abs/1706.06083.

    Parameters
    ---
    epsilon: float
        The maximum perturbation allowed

    step_size: float
        The step size of the gradient descent

    num_steps: int
        The number of steps to take in the gradient descent

    distance_metric: str
        The distance metric to use, either 'l2' or 'linf'
    """

    def __init__(self, model, args):
        super(NoAttack, self)

    def generate_attack(self,model, xs,ys):

        return xs.detach()


def get_attack(model, args):
    return NoAttack(model, args)
