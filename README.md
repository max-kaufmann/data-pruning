# Adversarial Corruptions
This is a repository for evaluating the robustness of image classifiers to unforseen adversaries, as described in the paper [Evaluating Robustness Against a Variety of Adversarial Corruptions](https://arxiv.org/).
## Installation
**TODO: Unsure what the install instructions should be**

## Usage

### Command-line evaluation of a model

The ``main.py`` script allows for the evaluation of a model against some specific attack e.g.:
```bash
python main.py --weights ../data/resnet_weights --architecture dm-wide-resnet --log ./log.jsonl --attack wood --epsilon medium
```
Where the hyperparameters are:

* *--weights*: String. Specifies the path to a ".pt" file, holding the state dictonary of the model.*
* *--architecture*: String. Specifies the name of the architecture used for evaluation.*
* *--attack*: Specifies the attack against which the model is being evaluated (See the [Attack README](./attacks/README.md)  for a long-form description of all of the attacks).
* *--dataset*: Specifies which dataset is being used to
* *--log*: Specifies the file to which the experiment results should be appended (see [Logging](#logging))/
* *--epsilon*: Specifies the strength of the attack. Can be either a float, or one of "low"m "medium" or "high" (which are mapped to the default values found in).
* *wood_num_rings*: An Attack-specific hyperparameter, there (See the [Attack README](./attacks/README.md))

 for a full list of hyperparameters adn their descriptions consult ``python main.py --help``

See [Extending the repository](#extending-the-repository) for more detail on what passing different arguments means.

The command line also allows for the tuning of other training hyperparameters, inclui and control of logging. For a full list of options, please run `python main.py --help`.

### Running a batch of experiments from a file

To more easily run a batch of experiments, main.py allows a list of experiments (and their hyperparameters) to be given in a jsonlines format.

e.g. given a list of experiments to be ran in a file called "batched_experiments.jsonl":
```javascript
{ "attack" : "wood","architecture":"resnet50", "epsilon": 0.1, ... }
{ "attack" : "snow","architecture":"resnet50","epsilon": 0.3 ... }
...
```
We sequentially run all experiments in the file:
```python
from main import run_experiment_file
run_experiment_file("./batched_experiments.jsonl")
```
### Logging

Logs use the [jsonlines](https://jsonlines.org/) format, with each JSON object corresponding to a single experiment. All experiment hyperparameters (i.e. all command line arguments) are recorded, as well as:
* Accuracy

* Avg. Loss

* Wall-clock experiment time (seconds)

* Proportion of datapoints for which loss increased after the attack (useful for debugging attacks). 

### Evaluating customized models

Customized models can be supported by [adding new architectures](#adding-new-architectures). Alternately, they can directly loaded and then evaluated in `main.py`. For instance, to evaluate the standard ImageNet resnet50 model, we can do something like this in `main.py`:

```python
from torchvision.models import resnet50, ResNet50_Weights


# This loads the standard imagenet resnet50 model
model = resnet50(weights="IMAGENET1K_V2")
# Notet that the ImageNetWrapper is required here to provide the proper normalization
model = ImageNetWrapper(model)
model.to(args.device)
model.eval()

# Define the data loader, note that the last step of the transform is "ToTensor";
# Can also be done using the get_test_dataset method, see the "Adding Datasets" section
test_dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
	
results = evaluate.evaluate(model, test_dataloader, attack, args)

```

## Extending the repository
To allow for easy extension of the repository, and to maintain clean code, all imports in  ``main.py``  are  dynamic. This means that the addition of new architectures/models/attacks to the repository is done by adding modules which implement the functonality expected by ``main.py``, which uses 


### Adding Attacks
When creating an attack ``attack_name``, which takes the hyperparameters ``hyperparameter_1, hyperparameter_2 ...`` we need to:

1. Create a new python module `attacks/attack_name.py`
2. Ensure that it implements a function called ``get_attack``, taking two arguments:
	* `model : nn.Module` This is the model for which the attack is bieng created
	* `args: argparse.Namespace` This is an object containing the parameters which were passed to ``main.py``  (accessed through the named attributes of the object, e.g.

    This should return a ``attacks.attacks.AttackInstance`` object which implements the required funconality.

An example of this can be found in ``attacks/wood.py``:

```python
class WoodAttack(attacks.attacks.AttackInstance):

    def __init__(self, model, args):
        super(WoodAttack, self).__init__(model, args)
        self.attack = WoodAdversary(args.epsilon, args.num_steps,
                                    args.step_size, args.wood_noise_resolution,
                                    args.wood_num_rings)

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return WoodAttack(model, args)
```

After an attack has been added, models can be evaluated against this attack by passing in the  ``--attack attack_name`` parameter to ``main.py``.

### Adding Datasets

Datasets are added by creating new packages in the ``/models/``  directory. To add a dataset called ``dataset_name``.

1. Create a new python package `models/dataset_name/`
2. Within that package, add a new module ``models/dataset_name/dataset_name.py``. 
3. Within that module, implement a function called ``get_test_datsaset``. This takes a single argument ``args`` ( a Namespace object containing the parameters passed into ``main.py``) and returns an object satisfying the PyTorch ``IterableDataset``interface.

A concrete example can be found in ``models/imagenet/imagenet.py``:

```python
import torchvision
import torchvision.transforms as transforms

import models.imagenet.imagenet_config as imagenet_config

test_transform =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

def get_test_dataset(args):
    test_dataset = torchvision.datasets.ImageNet(imagenet_config.imagenet_location,split="val",transform=test_transform)
    return test_dataset
```
**Note:** The last step of the test transform is "ToTensor", i.e., there is no normalization of the data.

### Adding New Architectures

 Model architectures are found in the package of their respective dataset. For example, when adding an architecture called ``architecture_name`` which functions on the dataset ``dataset_name``, we should:

1) Create a new module  ``models/dataset_name/architecture_name.py``  
2) Ensure the module implements a function ``get_model``, which takes in a single argument ``args``. This will be an `` argparse.Namespace``object containing the parameters passed to the program. 

The instance variable ``args.weights`` will contain string denoting the path to a saved ".pt" file, condtatining the state dictonary of a saved model.  This state dictonary should be loaded into the model, and the relevant ``torch.nn.Module``object should be returned. Care should be taken to load the model into the device specified in ``config.device``.

An example, slightly edited from ``models/cifar10/resnet50``:

```python
def get_model(args):
    model = ResNet(Bottleneck,[3, 4, 6, 3])
    model.load_state_dict(torch.load(weights,map_location=config.device))
    return model
```
