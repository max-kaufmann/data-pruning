import config

imagenet_location = "/data/hendrycks/imagenet/imagenet" # TODO: Must be a better way to do this than hardcoding it here 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

default_epsilons = {'none': [1, 1, 1], 'jpeg': [1/255, 3/255, 6/255], 'prison': [0.01, 0.03, 0.1], 'pgd': [1/255, 2/255, 8/255], 'wood': [0.03, 0.05, 0.1], 'elastic': [1/255, 2/255, 8/255], 'fbm': [0.03, 0.06, 0.3], 'whirlpool': [10, 40, 100], 'gabor': [0.02, 0.05, 0.1], 'pokadot': [1, 3, 5], 'klotski': [0.03, 0.1, 0.2], 'blur': [0.1, 0.3, 0.6], 'fog': [0.3, 0.5, 0.7],  'snow': [10, 15, 25], 'edge': [0.03, 0.1, 0.3], 'hsv': [0.01, 0.03, 0.05], 'mix': [5, 10, 40], 'texture': [0.01, 0.03, 0.2], 'glitch': [0.03, 0.1, 0.2], 'pixel': [5, 10, 20], 'kaleidescope': [0.03, 0.05, 0.1]}

