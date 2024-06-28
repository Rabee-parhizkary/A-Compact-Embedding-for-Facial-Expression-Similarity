import os

import numpy as np
import torch
import torch.nn as nn
# import torchvision.models as vision_model
# from models.densenet import DenseNet
import torchvision.transforms.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.distance import PairwiseDistance
from torch.utils import data
from tqdm import tqdm

import datas
import models.Cleaner
from eval_metrics import evaluate
from models.ModelDenseNet import DenseNet3


# Training
def train(epoch, net, use_cuda, trainloader, optimizer, criterion, model_name):
    print("train {} epoch".format(epoch))
    labels, distances = [], []
    triplet_loss_sum = 0.0
    net.train()
    num = len(trainloader)
    for batch_idx, (anc, pos, neg) in tqdm(enumerate(trainloader), total=num, desc=f'Epoch {epoch}'):
        if use_cuda:
            anc, pos, neg = anc.cuda(), pos.cuda(), neg.cuda()
        optimizer.zero_grad()
        anc, pos, neg = Variable(anc), Variable(pos), Variable(neg)
        anc_fea = net(anc)
        pos_fea = net(pos)
        neg_fea = net(neg)
        loss = criterion(anc_fea, pos_fea, neg_fea)
        # print(loss.item())
        loss.backward()
        optimizer.step()

        dists = l2_dist.forward(anc_fea, pos_fea)
        distances.append(dists.data.cpu().numpy())
        labels.append(np.ones(dists.size(0)))

        dists = l2_dist.forward(anc_fea, neg_fea)
        distances.append(dists.data.cpu().numpy())
        labels.append(np.zeros(dists.size(0)))
        triplet_loss_sum += loss.item()

    check_points_dir = ".\\check_points\\" + model_name
    check_points_path = os.path.join(check_points_dir, "triplet_v5_{}_checkpoint.pkl".format(epoch))

    # print(path)
    if not os.path.exists(check_points_dir):
        os.makedirs(check_points_dir)

    torch.save(net, check_points_path)
    avg_triplet_loss = triplet_loss_sum / trainset.__len__()
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    # print(labels)
    # print(distances)
    tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
    # print('  train set - Triplet Loss       = {:.8f}'.format(avg_triplet_loss))
    # print('  train set - Accuracy           = {:.8f}'.format(np.mean(accuracy)))

    log_dir = ".\\log\\" + model_name
    log_path = os.path.join(log_dir, "tune_v5_16.log")

    # Create the directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Open the log file and append the log entry
    with open(log_path, "a+") as log_file:
        log_file.write(
            "epoch: {0}, Triplet Loss: {1}, Accuracy: {2} \n".format(epoch, avg_triplet_loss, np.mean(accuracy))
        )


if __name__ == '__main__':
    models.Cleaner.clean_with_report()
    # torch.cuda.empty_cache()
    torch.multiprocessing.freeze_support()

    l2_dist = PairwiseDistance(2)
    use_cuda = torch.cuda.is_available()
    print('starting on cuda = ' + str(use_cuda))

    transform = transforms.Compose([  # transforms.Grayscale(),
        transforms.Resize((100, 100)), transforms.ToTensor()])

    coefficient_size = 0.1
    train_size_init = 12414
    test_size_init = 18954
    test_size = int(test_size_init * coefficient_size)
    train_size = int(train_size_init * coefficient_size)
    trainset = datas.fec_data.FecData(datas.fec_data.path_pd, transform, size=train_size)
    testset = datas.fec_data.FecData(datas.fec_data.path_test, transform, size=test_size)
    trainloader = data.DataLoader(trainset, batch_size=20, num_workers=0)
    testloader = data.DataLoader(testset, batch_size=20, num_workers=0)

    print('Finished loading data')

    net = DenseNet3(growth_rate=32, depth=30, reduction=0.5,
                    bottleneck=True, num_classes=16)

    if use_cuda:
        net.cuda()

    total_epoch = 20
    init_lr = 0.0005

    criterion = nn.TripletMarginLoss(margin=0.2)
    optimizer = optim.SGD(params=net.parameters(), lr=init_lr * 50, momentum=0.9, weight_decay=5e-4)

    for i in tqdm(range(total_epoch), desc='Main training'):
        train(i, net, use_cuda, trainloader, optimizer, criterion, model_name='Dense Net')
