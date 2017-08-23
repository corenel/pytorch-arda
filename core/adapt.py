"""Execute domain adaption for ARDA."""

import torch
from torch import nn

from misc import params
from misc.utils import (calc_gradient_penalty, get_inf_iterator, get_optimizer,
                        make_variable, save_model)


def train(classifier, generator, critic, src_data_loader, tgt_data_loader):
    """Train generator, classifier and critic jointly."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    classifier.train()
    generator.train()
    critic.train()

    # set criterion for classifier and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_c = get_optimizer(classifier, "Adam")
    optimizer_g = get_optimizer(generator, "Adam")
    optimizer_d = get_optimizer(critic, "Adam")

    # zip source and target data pair
    data_iter_src = get_inf_iterator(src_data_loader)
    data_iter_tgt = get_inf_iterator(tgt_data_loader)

    # counter
    g_step = 0

    # positive and negative labels
    pos_labels = make_variable(torch.FloatTensor([1]))
    neg_labels = make_variable(torch.FloatTensor([-1]))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        ###########################
        # 2.1 train discriminator #
        ###########################
        # requires to compute gradients for D
        for p in critic.parameters():
            p.requires_grad = True

        # set steps for discriminator
        if g_step < 25 or g_step % 500 == 0:
            # this helps to start with the critic at optimum
            # even in the first iterations.
            critic_iters = 100
        else:
            critic_iters = params.d_steps

        # loop for optimizing discriminator
        for d_step in range(critic_iters):
            # convert images into torch.Variable
            images_src, labels_src = next(data_iter_src)
            images_tgt, _ = next(data_iter_tgt)
            images_src = make_variable(images_src)
            labels_src = make_variable(labels_src.squeeze_())
            images_tgt = make_variable(images_tgt)
            if images_src.size(0) != params.batch_size or \
                    images_tgt.size(0) != params.batch_size:
                continue

            # zero gradients for optimizer
            optimizer_d.zero_grad()

            # compute source data loss for discriminator
            feat_src = generator(images_src)
            d_loss_src = critic(feat_src.detach())
            d_loss_src = d_loss_src.mean()
            d_loss_src.backward(neg_labels)

            # compute target data loss for discriminator
            feat_tgt = generator(images_tgt)
            d_loss_tgt = critic(feat_tgt.detach())
            d_loss_tgt = d_loss_tgt.mean()
            d_loss_tgt.backward(pos_labels)

            # compute gradient penalty
            gradient_penalty = calc_gradient_penalty(
                critic, feat_src.data, feat_tgt.data)
            gradient_penalty.backward()

            # optimize weights of discriminator
            d_loss = - d_loss_src + d_loss_tgt + gradient_penalty
            optimizer_d.step()

        ########################
        # 2.2 train classifier #
        ########################

        # zero gradients for optimizer
        optimizer_c.zero_grad()

        # compute loss for critic
        preds_c = classifier(generator(images_src).detach())
        c_loss = criterion(preds_c, labels_src)

        # optimize source classifier
        c_loss.backward()
        optimizer_c.step()

        #######################
        # 2.3 train generator #
        #######################
        # avoid to compute gradients for D
        for p in critic.parameters():
            p.requires_grad = False

        # zero grad for optimizer of generator
        optimizer_g.zero_grad()

        # compute source data classification loss for generator
        feat_src = generator(images_src)
        preds_c = classifier(feat_src)
        g_loss_cls = criterion(preds_c, labels_src)
        g_loss_cls.backward()

        # compute source data discriminattion loss for generator
        feat_src = generator(images_src)
        g_loss_src = critic(feat_src).mean()
        g_loss_src.backward(pos_labels)

        # compute target data discriminattion loss for generator
        feat_tgt = generator(images_tgt)
        g_loss_tgt = critic(feat_tgt).mean()
        g_loss_tgt.backward(neg_labels)

        # compute loss for generator
        g_loss = g_loss_src - g_loss_tgt + g_loss_cls

        # optimize weights of generator
        optimizer_g.step()
        g_step += 1

        ##################
        # 2.4 print info #
        ##################
        if ((epoch + 1) % params.log_step == 0):
            print("Epoch [{}/{}]:"
                  "d_loss={:.5f} c_loss={:.5f} g_loss={:.5f} "
                  "D(x)={:.5f} D(G(z))={:.5f} GP={:.5f}"
                  .format(epoch + 1,
                          params.num_epochs,
                          d_loss.data[0],
                          c_loss.data[0],
                          g_loss.data[0],
                          d_loss_src.data[0],
                          d_loss_tgt.data[0],
                          gradient_penalty.data[0]
                          ))

        #############################
        # 2.5 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            save_model(critic, "WGAN-GP_critic-{}.pt".format(epoch + 1))
            save_model(classifier,
                       "WGAN-GP_classifier-{}.pt".format(epoch + 1))
            save_model(generator, "WGAN-GP_generator-{}.pt".format(epoch + 1))

    return classifier, generator
