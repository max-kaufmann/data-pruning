import torch
import torch.nn.functional as F

from autoattack import AutoAttack
from attacks.attacks import AttackInstance
import config

class AutoAttackAdversary(AttackInstance):

    def __init__(self, args):
        super(AutoAttackAdversary, self).__init__(args)

        if args.distance_metric == "l2":
            norm = "L2"
        elif args.distance_metric == "linf":
            norm = "Linf"
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf', was {args.distance_metric}"
            )
        
        self.adversary = AutoAttack(None,
                        norm=norm,
                        eps=args.epsilon,
                        version='custom',
                        device = config.device,
                        verbose=False,
                        attacks_to_run = ['apgd-ce'])
        
        self.adversary.apgd.n_iter = args.num_steps

    def generate_attack(self,model,xs,ys):
        self.adversary.apgd.model = model
        self.adversary.model = model
        """
        Both of these model variables need to be updated because AutoAttack initialises 
        by also initialising an APGD class instance. Both these classes initialise with
        a `self.model = model` line but from then on from what I can see updating one will 
        not update the other.
        
        Perhaps creating an APGD instance directly could speed things up instead of going via
        AutoAttack, although I'm not sure if this aspects of the attack that we would want to keep.

        """

        xs, ys = xs.to(config.device), ys.to(config.device)
        x_adv = self.adversary.run_standard_evaluation(xs, ys, bs=len(ys))
        return x_adv


def get_attack(args):
    return AutoAttackAdversary(args)