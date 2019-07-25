from networks.caffenet import caffenet
from networks.mnist import lenet
from networks.resnet import resnet18, resnet50
from networks.alexnet import alexnet

nets_map = {
    'caffenet': caffenet,
    'alexnet': alexnet,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'lenet': lenet
}


def get_aux_net(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
