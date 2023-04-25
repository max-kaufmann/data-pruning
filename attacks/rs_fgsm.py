import torch
import torch.nn.functional as F

import attacks
from attacks.attacks import AttackInstance, tensor_clamp_l2
import config

class FGSM(AttackInstance):

    def __init__(self, args):
        super(FGSM, self).__init__(args)
        self.epsilon = args.epsilon
        self.step_size = args.step_size

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

    def generate_attack(self,model,xs,ys):

        xs, ys = xs.to(config.device), ys.to(config.device)
        delta = torch.zeros_like(xs)

        delta = delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        adv_inputs = (xs + delta)
        logits = model(adv_inputs)
        loss = F.cross_entropy(logits, ys)
        grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]

        adv_delta = self.step_size * torch.sign(grad)
        adv_delta = self.project_tensor(adv_delta, self.epsilon)
        adv_inputs = torch.clamp(xs + adv_delta, 0, 1)

        return adv_inputs.detach()


def get_attack(args):
    return FGSM(args)
