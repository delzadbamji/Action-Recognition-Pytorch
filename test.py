import pickle
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model
from network import R2Plus1D_model
# import display_action

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 20  # Number of epochs for training
resume_epoch = 20  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 10 # Run on test set every nTestInterval epochs
snapshot = 10 # Store a model every snapshot epochs
lr = 1e-3 # Learning rate
batch_size=1
num_worker=0

dataset = 'Original_video'


if dataset == 'Original_video':
    num_classes = 4
else:
    print('We only implemented Original_video datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]


if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'R2Plus1D' # Options: C3D or R2Plus1D
saveName = modelName + '-' + dataset
# saveName='R2Plus1D-Original_video'




def test_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval,batch_size=batch_size, num_worker=num_worker):
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    else:
        print('We only implemented C3D and R2plus1D models.')
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)


    test_data=VideoDataset(dataset=dataset, split='test', clip_len=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 num_workers=num_worker)


    test_size = len(test_dataloader.dataset)

    if useTest:
        model.eval()
        start_time = timeit.default_timer()

        running_loss = 0.0
        running_corrects = 0.0
        result=[]


        for i, (inputs, labels) in tqdm(enumerate(test_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.long()


            with torch.no_grad():
                outputs = model(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]

            videodict=test_data.getlabel2index()

            loss = criterion(outputs, labels)
            for j in range(len(preds)):
                result.append([str(preds[j].cpu().numpy()),str(labels[j].cpu().numpy()),test_data.fnames[i*batch_size+j]])

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)


        epoch_loss = running_loss / test_size
        epoch_acc = running_corrects.double() / test_size
        result=np.array(result)
        # np.savetxt("results.csv", result,delimiter=',')
        with open('result.pk', 'wb') as file:
            pickle.dump(result, file)

        writer.add_scalar('data/test_loss_epoch', epoch_loss, num_epochs)
        writer.add_scalar('data/test_acc_epoch', epoch_acc, num_epochs)

        print("[test] Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()

if __name__ == '__main__':
    test_model()