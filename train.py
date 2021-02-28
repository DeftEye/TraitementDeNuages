import sys
import glob
from tqdm import tqdm
import random

import torch
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import data
import models
import utils

"""
You can now stick together :
- your dataloader
- your model
- your losses
- an optimizer
- ModelCheckpointtensorboard callbacks

And write your training and testing function.
We have to backpropagate through two losses rather than just one. Hit the button, tweak your optimization/regularization settings and then go on.
"""
def load_tensors(path):
    """
        Load a collection of tensors that match the regular expression path
        The individual tensors are loaded in CPU

        Returns a concatenated tensors of all the loaded tensors.
    """
    tensors = None
    for tensor_filename in tqdm(glob.glob(path)):
        tensor = torch.load(tensor_filename, map_location='cpu')
        if not tensors:
            tensors = tensor
        else:
            for k in tensors:
                tensors[k] = torch.cat((tensors[k], tensor[k]), dim=0)
        del tensor
    for k in tensors:
        print("Key {} has shape {}".format(k, tensors[k].shape))
    return tensors


class LazyTensorsLoader(object):

    __tensor_filenames = []
    __indices = []
    __idx = 0
    __shuffle = False
    __tensor_keys = None

    def __init__(self, path, shuffle):
        self.__tensor_filenames = glob.glob(path)
        self.__indices = range(len(self.__tensor_filenames))
        self.__shuffle = shuffle

    def __iter__(self):
        # Random shuffle the idx
        if self.__shuffle:
            self.__indices = range(len(self.__tensor_filenames))
        self.__idx = 0
        return self

    def __len__(self):
        return len(self.__tensor_filenames)

    def __next__(self):
        if self.__idx >= len(self.__indices):
            raise StopIteration
        else:
            tensor_filename = self.__tensor_filenames[self.__idx]
            tensor = torch.load(tensor_filename, map_location='cpu')
            self.__idx += 1
            if not self.__tensor_keys:
                # We are in single object mode
                self.__tensor_keys= ["features", "bboxes", "labels"]
            return tuple(tensor[k] for k in self.__tensor_keys)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Whether to use GPU'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='The number of CPU threads used'
    )

    parser.add_argument(
        '--tensors',
        type=str,
        help='Where to find the input tensors. We expect <tensors>/train_xx.pt and <tensors>/valid_xx.pt',
        required=True
    )

    parser.add_argument(
        '--lowmem',
        action='store_true',
        help='Activate this option if on low mem GPU. If this option is selected, we will lazylly load the tensors. Otherwise we preload all the tensors. In lowmem mode, the minibatches are shuffled but there is no shuffle inside a minibatch, i.e. samples from different minibatches are never mixed in.'
    )

    parser.add_argument(
        '--logdir',
        type=str,
        help='Where to store logs',
        default='./logs'
    )

    args = parser.parse_args()
    num_classes = len(data.classes)
    epochs = 100

    # GPU or CPU ?
    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda')

    # Precomputed features loading
    if args.lowmem:
        train_loader = LazyTensorsLoader(args.tensors + "/train*.pt", shuffle=True)
        print("I found {} train tensors to load".format(len(train_loader)))
        valid_loader = LazyTensorsLoader(args.tensors + "/valid*.pt", shuffle=False)
        print("I found {} valid tensors to load".format(len(valid_loader)))

        one_batch_bboxes = next(iter(train_loader))[1] # features, bbox, label, ..

        if len(one_batch_bboxes.shape) == 2:
            target_mode = 'largest_bbox'
    

    one_batch = next(iter(train_loader))[0] # features, bbox, label, ..
    num_features = one_batch[0].numel()
    num_channels = one_batch[0].size()[0]
    print("{} features, {} channels".format(num_features, num_channels))

    ############################################ Model
    if(target_mode == 'largest_bbox'):
        model = models.SingleBboxHead(num_features, num_classes)

    ############################################ Optimize, LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    ############################################ Callbacks (Tensorboard, Checkpoint)
    # Log dir
    logdir = utils.generate_unique_logpath(args.logdir,target_mode)
    os.makedirs(logdir)

    print("The logs will be saved in {}".format(logdir))
    tensorboard_writer   = SummaryWriter(log_dir = logdir)
    model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt", model)

    num_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary_file = open(logdir + "/summary.txt", 'w')
    summary_text = """

Executed command
================
{}

Dataset
=======
PascalVOC preprocessed

Model summary
=============
    {}

{} trainable parameters

Optimizer
========
    {}
    """.format(" ".join(sys.argv),
               str(model).replace('\n','\n\t'),
               num_params,
               str(optimizer).replace('\n','\n\t'))
    summary_file.write(summary_text)
    summary_file.close()

    tensorboard_writer.add_text("Experiment summary", summary_text)
    print("{} parameters to be optimized".format(num_params))

    ############################################ Training loop

    if(target_mode == 'largest_bbox'):
        for t in range(epochs):
            print("Epoch {}".format(t))
            scheduler.step()

            train_reg_loss, train_acc = utils.train(model, train_loader, optimizer, device)

            val_reg_loss, val_acc = utils.test(model, valid_loader, device)
            print(" Validation : Bbox Loss : {:.4f}, Class Acc : {:.4f}".format(val_reg_loss, val_acc))

            tensorboard_writer.add_scalar('metrics/train/bbox_reg_loss', train_reg_loss, t)
            tensorboard_writer.add_scalar('metrics/train/class_acc',  train_acc, t)
            tensorboard_writer.add_scalar('metrics/val/bbox_reg_loss', val_reg_loss, t)
            tensorboard_writer.add_scalar('metrics/val/class_acc',  val_acc, t)
            model_checkpoint.update(val_reg_loss + (1.0 - val_acc))





