mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)
mean_wrn = (0.5, 0.5, 0.5)
std_wrn = (0.5, 0.5, 0.5)

default_epsilons={'none': [0.3, 0.3, 0.3], 'jpeg': [1/255, 3/255, 6/255], 'prison': [0.01, 0.1, 0.5], 'pgd': [2/255, 4/255, 16/255], 'wood': [0.01, 0.05, 0.1], 'elastic': [0.03125, 0.125, 0.25], 'fbm': [0.03, 0.1, 0.3], 'whirlpool': [20, 100, 200], 'gabor': [0.02, 0.05, 0.1], 'pokadot': [1, 3, 5], 'klotski': [0.03, 0.1, 0.2], 'blur': [0.3, 0.6, 1.0], 'fog': [0.2, 0.5, 1], 'lighting': [1, 3, 5], 'snow': [1, 3, 5], 'edge': [0.03, 0.2, 0.5], 'hsv': [0.03, 0.1, 0.2], 'mix': [1, 5, 10], 'texture': [0.03, 0.1, 0.3], 'glitch': [0.05, 0.1, 0.5], 'pixel': [1, 5, 10], 'kaleidescope': [0.03, 0.3, 1]}