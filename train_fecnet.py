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


# import imp

# l2_dist = PairwiseDistance(2)
# use_cuda = torch.cuda.is_available()
# print('starting on cuda = ' + str(use_cuda))
#
# transform = transforms.Compose([# transforms.Grayscale(),
#     transforms.Resize((224, 224)), transforms.ToTensor()])
#
# trainset = datas.fec_data.FecData(transform)
# testset = datas.fec_data.FecTestData(transform)
# trainloader = data.DataLoader(trainset, batch_size=24, num_workers=16)
# testloader = data.DataLoader(testset, batch_size=20, num_workers=16)
#
# print('Finished loading data')
#
# net = KitModel()
# # net = DenseNet(growthRate=12, depth=30, reduction=0.5,
# #                         bottleneck=True, nClasses=16)
#
# # net = visionmodel.densenet121(pretrained=True)
# # print(net)
# # net.classifier = nn.Linear(net.classifier.in_features, 16)
#
# if use_cuda:
#     net.cuda()
#
# total_epoch = 300
# init_lr = 0.0005
#
# criterion = nn.TripletMarginLoss(margin=0.2)
# optimizer = optim.SGD(params=net.parameters(), lr=init_lr * 50, momentum=0.9, weight_decay=5e-4)


# ignored_params = list(map(id, net.classifier.parameters()))
# base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
# optimizer = optim.SGD([
#     {'params': base_params},
#     {'params': net.classifier.parameters(), 'lr': init_lr*50}], lr=init_lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch, net, use_cuda, trainloader, optimizer, criterion):
    print("train {} epoch".format(epoch))
    labels, distances = [], []
    triplet_loss_sum = 0.0
    net.train()
    for batch_idx, (anc, pos, neg) in tqdm(enumerate(trainloader)):
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

    torch.save(net, "./check_points/triplet_v5_{}_checkpoint.pkl".format(epoch))
    avg_triplet_loss = triplet_loss_sum / trainset.__len__()
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    # print(labels)
    # print(distances)
    tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
    # print('  train set - Triplet Loss       = {:.8f}'.format(avg_triplet_loss))
    # print('  train set - Accuracy           = {:.8f}'.format(np.mean(accuracy)))

    with open("./log/tune_v5_16.log", "a+") as log_file:
        log_file.write(
            "epoch: {0}, Triplet Loss: {1}, Accuracy: {2} \n".format(epoch, avg_triplet_loss, np.mean(accuracy)))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.multiprocessing.freeze_support()

    l2_dist = PairwiseDistance(2)
    use_cuda = torch.cuda.is_available()
    print('starting on cuda = ' + str(use_cuda))

    transform = transforms.Compose([  # transforms.Grayscale(),
        transforms.Resize((224, 224)), transforms.ToTensor()])

    trainset = datas.fec_data.FecData(datas.fec_data.path_pd, transform)
    testset = datas.fec_data.FecData(datas.fec_data.path_test, transform)
    trainloader = data.DataLoader(trainset, batch_size=24, num_workers=0)
    testloader = data.DataLoader(testset, batch_size=20, num_workers=16)

    print('Finished loading data')

    net = DenseNet3(growth_rate=32, depth=120, reduction=0.5,
                    bottleneck=True, num_classes=16)

    # net = vision_model.densenet121(pretrained=True)
    # print(len(net))

    if use_cuda:
        net.cuda()

    total_epoch = 300
    init_lr = 0.0005

    criterion = nn.TripletMarginLoss(margin=0.2)
    optimizer = optim.SGD(params=net.parameters(), lr=init_lr * 50, momentum=0.9, weight_decay=5e-4)

    for i in range(1):
        train(i, net, use_cuda, trainloader, optimizer, criterion)

