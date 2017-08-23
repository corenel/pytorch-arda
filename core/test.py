"""Test result of domain adaption for ARDA."""

import torch.nn as nn

from misc.utils import make_variable


def test(classifier, generator, data_loader, dataset="MNIST"):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    generator.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels.squeeze_())

        preds = classifier(generator(images))
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {:.5f}, Avg Accuracy = {:2.5%}".format(loss, acc))
