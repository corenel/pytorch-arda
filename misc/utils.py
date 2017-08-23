"""Helpful functions for ARDA."""

import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable, grad

from datasets import get_mnist, get_usps
from misc import params


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def calc_gradient_penalty(D, real_data, fake_data):
    """Calculatge gradient penalty for WGAN-GP."""
    alpha = torch.rand(params.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = make_cuda(alpha)

    interpolates = make_variable(alpha * real_data + ((1 - alpha) * fake_data))
    interpolates.requires_grad = True

    disc_interpolates = D(interpolates)

    gradients = grad(outputs=disc_interpolates,
                     inputs=interpolates,
                     grad_outputs=make_cuda(
                         torch.ones(disc_interpolates.size())),
                     create_graph=True,
                     retain_graph=True,
                     only_inputs=True)[0]

    gradient_penalty = params.penalty_lambda * \
        ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def init_weights(layer):
    """Init weights for layers."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))


def get_optimizer(net, name="Adam"):
    """Get optimizer by name."""
    if name == "Adam":
        return optim.Adam(net.parameters(),
                          lr=params.learning_rate,
                          betas=(params.beta1, params.beta2))


def get_data_loader(name, train=True):
    """Get data loader by name."""
    if name == "MNIST":
        return get_mnist(train)
    elif name == "USPS":
        return get_usps(train)


def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels in data_loader:
            yield (images, labels)
