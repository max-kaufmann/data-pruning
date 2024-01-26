import torch
from torch.linalg import norm

def distance(images, model, args, num_classes=10, overshoot=0.02, max_iter=5):

    """
       :param images: batch of shape batch_size x depth x height x width
       :param model: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 5)
       :return: shortest distance to decision boundary for each image in batch
    """
    images = images.to(args.device)
    model = model.to(args.device)
    
    model_is_training = model.training
    model.eval()
    
    #manage indices
    initial_logits = model(images)
    I = initial_logits.argsort(dim=1,descending=True)
    I = I[:,0:num_classes]
    batch_size = len(I)
    batch_indices_remaining = torch.arange(batch_size).to(args.device)
    is_adv = torch.zeros_like(batch_indices_remaining, dtype=bool)
    
    #initialise variables
    image_dims = len(images.shape) - 1
    x = images.detach().clone()
    x.requires_grad = True
    r_tot = torch.zeros_like(x)
    loop_i = 0
    logits = model(x)
    distances = torch.zeros(batch_size).to(args.device)

    while batch_size > 0 and loop_i < max_iter: 
        #calculate f_prime
        sorted_logits = torch.gather(logits,1,I)
        f_prime = sorted_logits[:,1:] - sorted_logits[:,0:1]
        
        #calculate w
        F = f_prime.sum(axis=0)
        w = torch.stack([torch.autograd.grad(f_prime_k, x, retain_graph=True)[0][~is_adv] for f_prime_k in F],dim=1) 
        
        #find which classes have closest decision boundaries to each image in the batch
        p = abs(f_prime)/norm(w.view(batch_size,num_classes-1,-1),axis=2)
        mins, argmins = p.min(axis=1)

        #use that info to organise intermediate tensors by picking out and shaping relevant data from p and w
        p_mins = mins.view(batch_size,*[1]*image_dims)
        w_mins = w[list(range(batch_size)), argmins]

        #calculate pertubations r and update r_tot
        r = p_mins * w_mins / norm(w_mins.view(batch_size,-1),axis=1).view(batch_size,*[1]*image_dims)
        r_tot += r

        #apply pertubations
        x = images + (1+overshoot)*r_tot
        x = x.detach()
        x.requires_grad = True 
        logits = model(x)
        
        #assess whether classes have changed and drop out images that have changed class so they are no longer perturbed
        is_adv = logits.argmax(dim=1) != I[:,0]
        num_adv = is_adv.sum()
            
        if num_adv > 0:
            distances[batch_indices_remaining[is_adv]] = norm(r_tot[is_adv].view(num_adv,-1),axis=1)
            
            batch_indices_remaining = batch_indices_remaining[~is_adv]
            I = I[~is_adv]
            images = images[~is_adv]
            logits = logits[~is_adv]
            r_tot = r_tot[~is_adv]
            batch_size -= num_adv
           
        loop_i += 1

    if model_is_training:
        model.train()

    return distances.detach()