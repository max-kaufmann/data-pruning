import torch
import torch.nn.functional as F

import config
from attacks.attacks import AttackInstance, tensor_clamp_l2


class PGD(AttackInstance):
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

    def __init__(self, args):
        super(PGD, self).__init__( args)
        self.epsilon = args.epsilon
        self.step_size = args.step_size
        self.num_steps = args.num_steps

        if args.distance_metric == "l2":
            self.project_tensor = lambda x, epsilon: tensor_clamp_l2(
                x, 0, epsilon)
        elif args.distance_metric == "linf":
            self.project_tensor = lambda x, epsilon: torch.clamp(
                x, -epsilon, epsilon)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {args.distance_metric}"
            )

    def generate_attack(self,model, xs,ys):


        xs, ys = xs.to(config.device), ys.to(config.device)
        delta = torch.zeros_like(xs)

        delta = delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for i in range(0, self.num_steps):

            adv_inputs = (xs + delta)
            logits = model(adv_inputs)

            loss = F.cross_entropy(logits, ys)
            grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]

            delta = delta + self.step_size * torch.sign(grad)
            delta = self.project_tensor(delta, self.epsilon)
            delta = delta.detach()
            delta.requires_grad = True

        adv_inputs = torch.clamp(xs + delta, 0, 1)

        return adv_inputs.detach()


def get_attack(args):
    return PGD(args)
