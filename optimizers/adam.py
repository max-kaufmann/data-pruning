from torch.optim import Adam

def get_optimizer(model, args):
    optimizer = Adam(params=model.parameters(), lr=args.lr) 
    return optimizer