import torch
import torch.nn.functional as F

from autoattack import AutoAttack
from attacks.attacks import AttackInstance
import config

class AutoAttackAdversary(AttackInstance):

    def __init__(self, args):
        super(AutoAttackAdversary, self).__init__(args)
        self.epsilon = args.epsilon

        if args.distance_metric == "l2":
            self.norm = "L2"
        elif args.distance_metric == "linf":
            self.norm = "Linf"
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {args.distance_metric}"
            )

    def generate_attack(self,model,xs,ys):

        xs, ys = xs.to(config.device), ys.to(config.device)
        adversary = AutoAttack(model,
                               norm=self.norm,
                               eps=self.epsilon,
                               version='standard',
                               device = 'cpu',
                               verbose=False
                               )
        adversary.apgd.n_iter = 40
        adversary.attacks_to_run = ['apgd-ce']
        x_adv = adversary.run_standard_evaluation(xs, ys, bs=len(ys))
        return x_adv


def get_attack(args):
    return AutoAttackAdversary(args)