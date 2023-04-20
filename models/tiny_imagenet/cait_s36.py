import timm #TODO: Remove timm dependency
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import config
import models.tiny_imagenet.tiny_imagenet_config as tiny_imagenet_config
import models.tiny_imagenet.tiny_imagenet as tm

img_size = (384,384)

def get_model(weights,args):

    # TODO
    assert False
    model = timm.create_model('cait_s36_384', pretrained=True)
    model.reset_classifier(num_classes=200)
    tm.test_transform = transforms.Compose([tm.test_transform,transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)]) # Need to resize to 384x384, due to model input size

    if weights is not None:

        if weights.startswith('http'):
            state_dict = torch.hub.load_state_dict_from_url(weights,model_dir=config.project_path + "/models/tiny_imagenet/data/",map_location=config.device)['model_state_dict'] # TODO: Change to make it consistent with the rest of the loading
        else:
            state_dict = torch.load(weights)
        model.load_state_dict(state_dict)

    return model