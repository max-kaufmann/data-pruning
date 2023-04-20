import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from attacks.attacks import AttackInstance

# Taken from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
# which adapted from https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
# Explained here: https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf


def perlin_2d(gradients, res, shape):
    """
    This interpolates between random gradient vectors on a grid. Read https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf for 
    an explanation fo the algorithm.

    Paramters
    ---

    res (tuple): a tuple of the resolution for the x and y axis. res[0] must be a divisor of shape[0],
        same with res[1] and shape[1]. This describes the resolution of the random gradient vectors (Which are then interpolated between).

    angles (tensor): a tensor with size (res[0] + 1, res[1] + 1). Must be between 0 and 2*pi. This describes the directions of the random
    gradient vectors.
    """

    batch_size, _, _, _ = gradients.shape
    shape = (shape, shape)

    # The fade function controls the interpolation between the random gradient vectors, which preserves gradient and is zero at the required times.
    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    # For each coordinate position, we calculate how far away it is from the bottom left gradient vector.
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]),
                                      torch.arange(0, res[1], delta[1])),
                       dim=-1).repeat(batch_size, 1, 1, 1) % 1
    grid = grid.to(config.device)

    # Given the grid of gradients of size (resolution,resolution) we offset the grid and tile it to the size of the inputs
    def tile_grads(slice1, slice2):
        return gradients[:, slice1[0]:slice1[1], slice2[0]:slice2[1],
                         ...].repeat_interleave(d[0], 1).repeat_interleave(
                             d[1], 2).to(config.device)

    # When doing the perlin noise algorithm, we need to calculate the vectors from the nearest coordinate points to the current point.
    # This is done by subtracting 1 from either one of the u,v coordinates. (see https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf)
    def dot(grad, shift):
        return (torch.stack(
            (grid[:, :shape[0], :shape[1], 0] + shift[0],
             grid[:, :shape[0], :shape[1], 1] + shift[1]),
            dim=-1) * grad[..., :shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0]) # Gradient-normalsied distance from the bottom left
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0]) # Gradient-normalised distance from the bottom right
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1]) # Gradient-normalised distance from the top right
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1]) # Gradient-normalised distance from the top left
    t = fade(grid[:, :shape[0], :shape[1]]) # This fade functoin decides how we interpolate between the grid points.

    return math.sqrt(2) * torch.lerp(torch.lerp(
        n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]) # Do a bi-linear interpolatoin of the 4 grid points


def fractional_brownian_motion(fbm_vars, octaves, image_shape):

    batch_size, num_channels, height, width = image_shape
    nearest_power_of_2 = int(math.pow(2, math.ceil(math.log(height, 2))))
    padding = (nearest_power_of_2 - height) // 2
    f = 1  # frequency
    a = 1.0  # amplitude
    t = torch.zeros(batch_size,
                    nearest_power_of_2,
                    nearest_power_of_2,
                    device=config.device)

    for i in range(octaves):
        t += a * perlin_2d(fbm_vars[i], (8 * f, 8 * f), nearest_power_of_2)
        f *= 2
        a *= 0.5

    if padding != 0:
        t = t[:, padding:-padding, padding:-padding]

    return t.unsqueeze(1)


def fbm_creator(fbm_vars, image, octaves):
    noise = fractional_brownian_motion(fbm_vars, octaves, image.shape)

    return (image + noise).clamp(0, 1)


class FbmAdversary(nn.Module):
    """A class implementing fractional brownian motion, which is made up of several layers of what is called "perlin noise" (https://web.archive.org/web/20160530124230/http://freespace.virgin.net/hugo.elias/models/m_perlin.htm)
     each at a different amplitude and frequency (each of these is called an octave). Perlin noise is generated by sampling random gradient vectors in a grid, and then interpolating a cubic polynomial between each of these
     grid points.

     Parameters
     ----------
        Scale: float
            Controls by how much the noise is allowed to be scaled

        num_steps: int
            The number of optimisation steps taken within the inner loop of the attack

        step_size: float
            The size of each individual optimisation step taken in the attack.


        """

    def __init__(self, num_steps, step_size, epsilon):
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def forward(self, model, inputs, targets):
        """
        :param model: the classifier's forward method
        :param inputs: batch of images
        :param targets: true labels
        :return: perturbed batch of images
        """

        # create initial variables
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        batch_size, channels, height, width = inputs.shape
        size = inputs.shape[-1]
        bits = math.ceil(math.log(size, 2))
        octaves = bits - 3
        fbm_vars = []

        for i in range(octaves):
            size = 8 * 2**i + 1
            var = self.epsilon * \
                torch.rand((batch_size, size, size, 2), device=config.device)
            var.requires_grad_()
            fbm_vars.append(var)

        # begin optimizing the inner loop.
        for i in range(self.num_steps):

            adv_inputs = fbm_creator(fbm_vars, inputs.detach(), octaves)
            logits = model((adv_inputs))
            loss = F.cross_entropy(logits, targets)

            grads = torch.autograd.grad(loss, fbm_vars)
            # clamp variables
            for i in range(octaves):
                fbm_vars[i] = fbm_vars[i] + \
                    torch.sign(grads[i]) * self.step_size
                torch.clamp(fbm_vars[i], -self.epsilon, self.epsilon)
                fbm_vars[i] = fbm_vars[i].detach()
                fbm_vars[i].requires_grad_()

        adv_inputs = (adv_inputs).clamp(0, 1)
        return adv_inputs


class FbmAttack(AttackInstance):

    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = FbmAdversary(args.num_steps, args.step_size,
                                   args.epsilon)
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return FbmAttack(model, args)
