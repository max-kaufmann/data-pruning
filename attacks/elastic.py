import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import config
from attacks.attacks import AttackInstance

base_flow = None


def get_base_flow(shape):

    _, _, height, width = shape
    xflow, yflow = torch.meshgrid(torch.linspace(-1,
                                                 1,
                                                 height,
                                                 device=config.device),
                                  torch.linspace(-1,
                                                 1,
                                                 width,
                                                 device=config.device),
                                  indexing="xy")
    base_flow = torch.stack((xflow, yflow), dim=-1)
    base_flow = torch.unsqueeze(base_flow, dim=0)

    return base_flow


def flow(image, flow_variables, kernel_size, sigma):

    flow_variables_x_blurred = transforms.functional.gaussian_blur(
        flow_variables[..., 0], kernel_size=kernel_size, sigma=sigma)
    flow_variables_y_blurred = transforms.functional.gaussian_blur(
        flow_variables[..., 1], kernel_size=kernel_size, sigma=sigma)

    flow_variables = torch.stack(
        (flow_variables_x_blurred, flow_variables_y_blurred), dim=-1)

    if not hasattr(flow,"base_flow"):
        flow.base_flow = get_base_flow(image.shape)

    flow_image = F.grid_sample(image,
                               flow.base_flow + flow_variables,
                               mode='bilinear').requires_grad_()

    return flow_image




class ElasticAdversary(nn.Module):

    def __init__(self, epsilon, num_steps, step_size, kernel_size, kernel_std):
        super().__init__()
        '''
        Implementaiton of the Elastic attack, functoning by perturbing the image using an optimisable flow field.

        Parameters
        ----
        
            epsilon (float): maximum perturbation
            num_steps (int): number of steps
            step_size (float): step size
            kernel_size (int): kernel size for gaussian blur of the variables
            kernel_std (float): standard deviation for gaussian blur of the variables
        '''
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.step_size = step_size
        self.kernel_size = kernel_size
        self.kernel_std = kernel_std

    def forward(self, model, inputs, targets):

        batch_size, _, height, width = inputs.size()
        inputs.requires_grad = True

        epsilon = self.epsilon
        step_size = self.step_size

        flow_vars = epsilon * (torch.rand((batch_size, height, width, 2),
                                          requires_grad=True,
                                          device=config.device) * 2 - 1)



        for _ in range(self.num_steps):

            adv_inputs = flow(inputs, flow_vars, self.kernel_size,
                              self.kernel_std)
            outputs = model(adv_inputs)
            loss = F.cross_entropy(outputs, targets)

            grad = torch.autograd.grad(loss, flow_vars)[0]

            flow_vars = flow_vars + step_size * torch.sign(grad)
            flow_vars = torch.clamp(flow_vars, -epsilon, epsilon).detach()
            flow_vars.requires_grad = True

        adv_inputs = flow(inputs, flow_vars, self.kernel_size, self.kernel_std)

        return adv_inputs

class ElasticAttack(AttackInstance):

    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = ElasticAdversary(args.epsilon, args.num_steps,
                                       args.step_size,
                                       args.elastic_kernel_size,
                                       args.elastic_kernel_sigma)

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return ElasticAttack(model, args)
