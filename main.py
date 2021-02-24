import argparse
import time
import os
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
from model import model_dict


# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--model', type=str, default='deit_tiny',
                    help='name of the model to train: to choose emong: resnext, deit_tiny and deit_bese')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='outputs/models/', metavar='E',
                    help='folder where experiment outputs are located.')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)


# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)


# Data initialization and loading
from data import data_transforms

dsets = {x: datasets.ImageFolder(os.path.join(args.data, x), data_transforms[x])
         for x in ['train_images', 'val_images']}

dset_sizes = {x: len(dsets[x]) for x in ['train_images', 'val_images']}

dataloaders = {"train_images": torch.utils.data.DataLoader(dsets["train_images"], batch_size=args.batch_size,
                                              shuffle=True, num_workers=6),
              "val_images":torch.utils.data.DataLoader(dsets["val_images"], batch_size=args.batch_size,
                                              shuffle=False, num_workers=6)}



# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

model = model_dict[args.model]
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# training
def train(model, criterion, dataloaders, dataset_sizes,  optimizer, scheduler, num_epochs=10):
    """train method
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train_images', 'val_images']:
            if phase == 'train_images':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            current_loss = 0.0
            current_corrects = 0

            # Here's where the training happens
            print('Iterating through data...')

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # We need to zero the gradients, don't forget it
                optimizer.zero_grad()

                # Time to carry out the forward training poss
                # We only need to log the loss stats if we are in training phase
                with torch.set_grad_enabled(phase == 'train_images'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train_images':
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == 'val_images' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)


train(model, criterion, dataloaders,dset_sizes,optimizer, exp_lr_scheduler, num_epochs= args.epochs)
model_file = args.experiment + '/model_' + str(args.epochs) + '.pth'
torch.save(model.state_dict(), model_file)
print('Best model saved to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
