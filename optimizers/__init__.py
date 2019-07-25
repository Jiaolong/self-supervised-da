from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}

def get_optimizer(cfg):
    if cfg is None:
        print("Using SGD optimizer")
        return SGD

    else:
        opt_name = cfg["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        print("Using {} optimizer".format(opt_name))
        return key2opt[opt_name]
