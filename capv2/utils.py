#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:21:41 2018

@author: lu
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision.utils as vutils
import argparse


# Normalize MNIST dataset.
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def one_hot_encode(target, length):
    """Converts batches of class indices to classes of one-hot vectors."""
    batch_s = target.size(0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


def checkpoint(state, epoch):
    """Save checkpoint"""
    model_out_path = 'results/trained_model/model_epoch_{}.pth'.format(epoch)
    torch.save(state, model_out_path)
    print('Checkpoint saved to {}'.format(model_out_path))


def load_mnist(args):
    """Load MNIST dataset.
    The data is split and normalized between train and test sets.
    """
    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cuda else {}

    print('===> Loading training datasets')
    # MNIST dataset
    training_set = datasets.MNIST(
        './data', train=True, download=True, transform=data_transform)
    # Input pipeline
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading testing datasets')
    testing_set = datasets.MNIST(
        './data', train=False, download=True, transform=data_transform)
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return training_data_loader, testing_data_loader


def squash(sj, dim=2):
    """
    The non-linear activation used in Capsule.
    It drives the length of a large vector to near 1 and small vector to 0

    This implement equation 1 from the paper.
    """
    sj_mag_sq = torch.sum(sj**2, dim, keepdim=True)
    #print("-------0--sj_mag_sq-----")
    #print(type(sj))
    #print(sj)
    #print("-------1--sj_mag_sq-----")
    #print(sj_mag_sq)
    #print("-------2--sj_mag_sq-----")
    # ||sj||
    sj_mag = torch.sqrt(sj_mag_sq)
    v_j = (sj_mag_sq / (1.0 + sj_mag_sq)) * (sj / sj_mag)
    #print('1-----------------v_j--------------------------')
    #print(v_j)
    #print('2-----------------v_j--------------------------')
    #print('vj: ', v_j)
    #v_j = sj
    return v_j


def margin_loss(input, target):
    """
    Class loss

    Implement equation 4 in section 3 'Margin loss for digit existence' in the paper.

    Args:
        input: [batch_size, 10, 16, 1] The output from `DigitCaps` layer.
        target: target: [batch_size, 10] One-hot MNIST labels.

    Returns:
        l_c: A scalar of class loss or also know as margin loss.
    """
    batch_size = input.size(0)

    # ||vc|| also known as norm.
    v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

    # Calculate left and right max() terms.
    zero = Variable(torch.zeros(1))
    #if self.cuda_enabled:
    #    zero = zero.cuda()
    m_plus = 0.9
    m_minus = 0.1
    loss_lambda = 0.5
    max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)**2
    max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)**2
    t_c = target
    # Lc is margin loss for each digit of class c
    l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
    l_c = l_c.sum(dim=1)

    return l_c

def mask(out_digit_caps, cuda_enabled=True):
    """
    In the paper, they mask out all but the activity vector of the correct digit capsule.

    This means:
    a) during training, mask all but the capsule (1x16 vector) which match the ground-truth.
    b) during testing, mask all but the longest capsule (1x16 vector).

    Args:
        out_digit_caps: [batch_size, 10, 16] Tensor output of `DigitCaps` layer.

    Returns:
        masked: [batch_size, 10, 16, 1] The masked capsules tensors.
    """
    # a) Get capsule outputs lengths, ||v_c||
    v_length = torch.sqrt((out_digit_caps**2).sum(dim=2))

    # b) Pick out the index of longest capsule output, v_length by
    # masking the tensor by the max value in dim=1.
    _, max_index = v_length.max(dim=1)
    max_index = max_index.data

    # Method 1: masking with y.
    # c) In all batches, get the most active capsule
    # It's not easy to understand the indexing process with max_index
    # as we are 3D animal.
    batch_size = out_digit_caps.size(0)
    masked_v = [None] * batch_size # Python list
    for batch_ix in range(batch_size):
        # Batch sample
        sample = out_digit_caps[batch_ix]

        # Masks out the other capsules in this sample.
        v = Variable(torch.zeros(sample.size()))
        if cuda_enabled:
            v = v.cuda()

        # Get the maximum capsule index from this batch sample.
        max_caps_index = max_index[batch_ix]
        v[max_caps_index] = sample[max_caps_index]
        masked_v[batch_ix] = v # append v to masked_v

    # Concatenates sequence of masked capsules tensors along the batch dimension.
    masked = torch.stack(masked_v, dim=0)

    return masked


def save_image(image, file_name):
    """
    Save a given image into an image file
    """
    # Check number of channels in an image.
    if image.size(1) == 1:
        # Grayscale
        image_tensor = image.data.cpu() # get Tensor from Variable

    vutils.save_image(image_tensor, file_name)


def softmax(input, dim=1):
    """
    nn.functional.softmax does not take a dimension as of PyTorch version 0.2.0.

    This was created to add dimension support to the existing softmax function
    for now until PyTorch 0.4.0 stable is release.

    GitHub issue tracking this: https://github.com/pytorch/pytorch/issues/1020

    Arguments:
        input (Variable): input
        dim (int): A dimension along which softmax will be computed.
    """
    input_size = input.size()

    trans_input = input.transpose(dim, len(input_size) - 1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)

    return soft_max_nd.transpose(dim, len(input_size) - 1)


def accuracy(output, target, cuda_enabled=True):
    """
    Compute accuracy.

    Args:
        output: [batch_size, 10, 16, 1] The output from DigitCaps layer.
        target: [batch_size] Labels for dataset.

    Returns:
        accuracy (float): The accuracy for a batch.
    """
    batch_size = target.size(0)
    #print('In acc----------------------------batch_size:', batch_size)
    #print('In acc----------------------------output:', output)
    v_length = torch.sqrt((output**2).sum(dim=2, keepdim=True))
    #print('In acc----------------------------v_length:', v_length)
    softmax_v = softmax(v_length, dim=1)
    assert softmax_v.size() == torch.Size([batch_size, 2, 1, 1])
    #print('In acc----------------------------softmax_v:', softmax_v)

    _, max_index = softmax_v.max(dim=1)
    assert max_index.size() == torch.Size([batch_size, 1, 1])
    #print('In acc----------------------------max_index:', max_index)

    pred = max_index.squeeze() #max_index.view(batch_size)
    assert pred.size() == torch.Size([batch_size])
    print('In acc----------------------------pred:', pred)

    if cuda_enabled:
        target = target.cuda()
        pred = pred.cuda()
    #print(type(target), type(pred.data))
    correct_pred = torch.eq(target, pred.data) # tensor
    #print('In acc----------------------------correct_pred:', correct_pred)
    # correct_pred_sum = correct_pred.sum() # scalar. e.g: 6 correct out of 128 images.
    acc = correct_pred.float().mean() # e.g: 6 / 128 = 0.046875

    return acc


def to_np(param):
    """
    Convert values of the model parameters to numpy.array.
    """
    return param.clone().cpu().data.numpy()


def str2bool(v):
    """
    Parsing boolean values with argparse.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
