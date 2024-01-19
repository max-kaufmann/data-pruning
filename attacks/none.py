import torch
import torch.nn.functional as F

import attacks
import config
from attacks.attacks import AttackInstance


class NoAttack(AttackInstance):

    def __init__(self):
        super(NoAttack, self)

    def generate_attack(self, model, xs, ys):

        return xs.detach()


def get_attack(args):
    return NoAttack()
