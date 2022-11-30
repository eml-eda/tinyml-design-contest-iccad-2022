import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchinfo import summary
import torchvision.transforms as transforms

import augmentation
import models
import losses
from utils import ToTensor, IEGM_DataSET, stats_report


def train_func(epoch_num, net, trainloader, testloader, criterion,
               optimizer, scheduler, device):

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(epoch_num):

        net.train()
        net = net.float().to(device)

        running_task_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # outputs = net(inputs).squeeze(-1)
            outputs = net(inputs)
            task_loss = criterion(outputs, labels)
            task_loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / inputs.size(0)
            correct = 0.0

            running_task_loss += task_loss.item()
            i += 1
        scheduler.step()

        print(f'[Epoch, Batches] is [{epoch+1}, {i}]')
        print(f'Train Acc: {accuracy / i:.2f}', end='\t')
        print(f'Train loss: {running_task_loss / i:.2E}')

        train_loss.append(running_task_loss / i)
        train_acc.append((accuracy / i).item())

        accuracy = 0.0
        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0

        for data_test in testloader:
            net.eval()
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            # outputs_test = net(IEGM_test).squeeze(-1)
            outputs_test = net(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        print(f'Test Acc: {correct / total:.2f}', end='\t')
        print(f'Test Loss: {running_loss_test / i:.2E}')

        test_loss.append(running_loss_test / i)
        test_acc.append((correct / total).item())

    return (train_loss, train_acc), (test_loss, test_acc)


def test_func(net, testloader, device):
    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0

    for data_test in testloader:
        net.eval()
        inputs_test, labels_test = data_test['IEGM_seg'], data_test['label']
        seg_label = copy.deepcopy(labels_test)

        inputs_test = inputs_test.float().to(device)
        labels_test = labels_test.to(device)

        # outputs_test = net(inputs_test).squeeze(-1)
        outputs_test = net(inputs_test)
        _, predicted_test = torch.max(outputs_test.data, 1)

        if seg_label == 0:
            segs_FP += (labels_test.size(0) -
                        (predicted_test == labels_test).sum()).item()
            segs_TN += (predicted_test == labels_test).sum().item()
        elif seg_label == 1:
            segs_FN += (labels_test.size(0) -
                        (predicted_test == labels_test).sum()).item()
            segs_TP += (predicted_test == labels_test).sum().item()

    return segs_TP, segs_TN, segs_FP, segs_FN


def main(args):
    # Training Hyperparameters
    BATCH_SIZE = args.bs
    BATCH_SIZE_TEST = 1
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    model_name = str(args.model)
    now = time.strftime("%Y%m%d-%H%M%S")
    path_data = args.path_data
    path_indices = args.path_indices
    saving_path = Path(args.saving_path) / model_name / now
    saving_path.mkdir(parents=True)

    # Save CLI Args
    file = open(saving_path / 'cli_args.txt', 'w')
    file.write(str(args))
    file.close()

    # Instantiating NN
    net = models.__dict__[model_name](args.size)

    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    if args.augment:
        # Gather data from trainset
        x = []
        y = []
        for sample in trainset:
            x.append(sample['IEGM_seg'])
            y.append(sample['label'])
        x = np.expand_dims(np.vstack(x), axis=1)
        y = np.vstack(y)

        # Augment data
        augmentations = [
            augmentation.Jittering(1.0, 0.05),
            augmentation.Scaling(1.0, 0.05),
            augmentation.MagWarp(1.0, 0.5, 4),
            augmentation.TimeWarp(1.0, 0.5, 4),
            augmentation.Jittering(1.0, 0.2),
            augmentation.MagWarp(1.0, 0.8, 4),
            augmentation.TimeWarp(1.0, 0.2, 4),
            augmentation.Scaling(1.0, 0.2),
        ]
        augmenter = augmentation.DataAugment(x, y, augments=augmentations)
        x_aug, y_aug = augmenter.run()
        trainset = augmentation.AugmentedData(x_aug, y_aug)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True,
                            num_workers=0)

    print("Training Dataset loading finish.")
    input_shape = trainset[0]['IEGM_seg'].shape
    print("Input Shape:", input_shape)
    net_summary = summary(net, input_size=(1,) + input_shape)
    print(f'Size: {net_summary.total_params}')
    print(f'MACs: {net_summary.total_mult_adds}')

    epoch_num = EPOCH
    criterion = losses.soft_fb_loss(beta=2)
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[12, 24],
                                               gamma=0.2)
    tr_res, ts_res = train_func(epoch_num, net, trainloader,
                                testloader, criterion, optimizer, scheduler,
                                device)
    train_loss, train_acc = tr_res
    test_loss, test_acc = ts_res
    torch.save(net, saving_path / 'model.pkl')

    file = open(saving_path / 'loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(test_acc))
    file.write('\n\n')
    file.close()

    # Final Test Phase
    segs_TP, segs_TN, segs_FP, segs_FN = test_func(net, testloader, device)
    file = open(saving_path / 'test_stats.txt', 'w')
    file.write('segments: TP, FN, FP, TN\n')
    output_segs = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
    file.write(output_segs + '\n')
    file.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model',
                           help=f'Model Name. One of: {models.__all__}')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2)
    argparser.add_argument('--lr', type=float, help='learning rate',
                           default=0.0001)
    argparser.add_argument('--bs', type=int,
                           help='total  bsfor traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./data/')
    argparser.add_argument('--path_indices', type=str,
                           default='./data_indices/')
    argparser.add_argument('--saving_path', type=str,
                           default='./saved_models')
    argparser.add_argument('--augment', default=False, action='store_true')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    print("device is --------------", device)

    main(args)
