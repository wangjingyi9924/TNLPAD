import argparse
import copy
import datetime
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import data_load
import resnet
import tools
from lenet import LetNet_Decomposition

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=0, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='output/result/')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.2)
parser.add_argument('--noise_type', type=str, help='[symmetric, asymmetric, pairflip, instance]', default='symmetric')
parser.add_argument('--dataset', type=str, help='[mnist, fmnist, cifar10, cifar100]', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=350)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--weight_decay', type=float, help='l2', default=0.001)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
parser.add_argument('--train_len', type=int, help='the number of training data', default=54000)
parser.add_argument('--c1', type=float, help='hyper-parameter c1', default=0.0001)
parser.add_argument('--c2', type=float, help='hyper-parameter c2', default=1)
parser.add_argument('--tips', type=str, help='tips', default='TNLPAD')
args = parser.parse_args()

print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

learning_rate = args.lr

# Load dataset
def load_data(args):

    if args.dataset == 'fmnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 100
        args.batch_size = 32
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.fmnist_dataset(True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,)), ]),
                                                 target_transform=tools.transform_target,
                                                 dataset=args.dataset,
                                                 noise_type=args.noise_type,
                                                 noise_rate=args.noise_rate,
                                                 split_per=args.split_percentage,
                                                 random_seed=args.seed)

        val_dataset = data_load.fmnist_dataset(False,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,)), ]),
                                               target_transform=tools.transform_target,
                                               dataset=args.dataset,
                                               noise_type=args.noise_type,
                                               noise_rate=args.noise_rate,
                                               split_per=args.split_percentage,
                                               random_seed=args.seed)

        test_dataset = data_load.fmnist_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)), ]),
            target_transform=tools.transform_target)

    if args.dataset == 'mnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 100
        args.batch_size = 32
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.mnist_dataset(True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)
        val_dataset = data_load.mnist_dataset(False,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,)),
                                              ]),
                                              target_transform=tools.transform_target,
                                              dataset=args.dataset,
                                              noise_type=args.noise_type,
                                              noise_rate=args.noise_rate,
                                              split_per=args.split_percentage,
                                              random_seed=args.seed)

        test_dataset = data_load.mnist_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
            target_transform=tools.transform_target)

    if args.dataset == 'cifar10':
        args.channel = 3
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 100
        args.batch_size = 64
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar10_dataset(True,
                                                  transform=transforms.Compose([
                                                      transforms.RandomCrop(32, padding=4),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                           (0.2023, 0.1994, 0.2010)),
                                                  ]),
                                                  target_transform=tools.transform_target,
                                                  dataset=args.dataset,
                                                  noise_type=args.noise_type,
                                                  noise_rate=args.noise_rate,
                                                  split_per=args.split_percentage,
                                                  random_seed=args.seed)

        val_dataset = data_load.cifar10_dataset(False,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)

        test_dataset = data_load.cifar10_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            target_transform=tools.transform_target)

    if args.dataset == 'cifar100':
        args.channel = 3
        args.num_classes = 100
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 100
        args.batch_size = 64
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar100_dataset(True,
                                                   transform=transforms.Compose([
                                                       transforms.RandomCrop(32, padding=4),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2023, 0.1994, 0.2010)),
                                                   ]),
                                                   target_transform=tools.transform_target,
                                                   dataset=args.dataset,
                                                   noise_type=args.noise_type,
                                                   noise_rate=args.noise_rate,
                                                   split_per=args.split_percentage,
                                                   random_seed=args.seed)

        val_dataset = data_load.cifar100_dataset(False,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010)),
                                                 ]),
                                                 target_transform=tools.transform_target,
                                                 dataset=args.dataset,
                                                 noise_type=args.noise_type,
                                                 noise_rate=args.noise_rate,
                                                 split_per=args.split_percentage,
                                                 random_seed=args.seed)

        test_dataset = data_load.cifar100_test_dataset(
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            target_transform=tools.transform_target)

    return train_dataset, val_dataset, test_dataset


save_dir = args.result_dir + args.dataset + '/' + args.noise_type + '/' + str(
    args.noise_rate) + '/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = logit.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def weight_flatten_by_name(model, name):
    params = []
    for u in model.named_parameters():
        if name in u[0]:
            params.append(u[1].view(-1))
    params = torch.cat(params)
    return params

def g1(x):
    return (x+1)*args.c1

def g2(x):
    return 1 / ((x+1)**(args.c2))

# Training code
def train(train_loader, epoch, model, optimizer1, args, criterion):
    model.train()
    train_total = 0
    train_correct = 0
    # Record the model from the previous epoch.
    pre_model = copy.deepcopy(model)

    for i, (data, labels, indexes) in enumerate(train_loader):

        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        logits1 = model(data)

        loss = criterion(logits1, labels.long())

        sigma_t = weight_flatten_by_name(model, 'sigma')
        sigma_t_pre = weight_flatten_by_name(pre_model, 'sigma')
        gamma = weight_flatten_by_name(model, 'gamma')
        delta_sigma = torch.norm((sigma_t - sigma_t_pre), p=2)
        gamma_norm = torch.norm(gamma, p=2)
        loss1 = g1(epoch) * delta_sigma
        loss2 = g2(epoch) * gamma_norm

        loss = loss + loss1 + loss2

        loss.backward()
        optimizer1.step()

        optimizer1.zero_grad()

        acc = accuracy(logits1, labels, topk=(1,))
        prec1 = float(acc[0])
        train_total += 1
        train_correct += prec1

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                  % (epoch + 1, args.n_epoch, i + 1, args.train_len // args.batch_size, prec1, loss.item()))

    train_acc1 = float(train_correct) / float(train_total)
    return train_acc1

# Validate the model on the validation set.
def evaluate(val_loader, model1):
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for data, labels, _ in val_loader:

            if torch.cuda.is_available():
                data = data.cuda()

            logits1 = model1(data)
            _, pred1 = torch.max(logits1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1

# Validate the model on the test set.
def evaluate_test(test_loader, model1):
    model1.eval()  # Change model to 'eval' mode.
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, _ in test_loader:

            if torch.cuda.is_available():
                data = data.cuda()

            logits = model1.predict(data)
            total += labels.size(0)

            _, pred = torch.max(logits.data, 1)
            correct += (pred.cpu() == labels.long()).sum()


        acc2 = 100 * float(correct) / float(total)

    return acc2


def main(args):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    txtfile = save_dir + ".txt"
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

    # Data Loader (Input Pipeline)
    print('Loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)

    # Define models
    print('Building model...')

    if args.dataset == 'mnist':
        clf1 = LetNet_Decomposition()
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'fmnist':
        clf1 = resnet.ResNet18(input_channel=1, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'cifar10':
        clf1 = resnet.ResNet18(input_channel=3, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[40, 80], gamma=0.1)
    elif args.dataset == 'cifar100':
        clf1 = resnet.ResNet50(input_channel=3, num_classes=100)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[40, 80], gamma=0.1)

    if torch.cuda.is_available():
        clf1.cuda()

    with open(txtfile, "a") as myfile:
        myfile.write('epoch  train_acc   val_acc   test_sigma_acc  \n')

    epoch = 0
    train_acc = 0

    val_acc = evaluate(val_loader, clf1)
    print('Epoch [%d/%d] Val Accuracy on the %s val data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(val_dataset), val_acc))

    test_sigma_acc = evaluate_test(test_loader, clf1)
    print('Epoch [%d/%d] Test Sigma Accuracy on the %s test data: Model1 %.4f %% ' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_sigma_acc))

    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(
            str(int(epoch)) + ' ' + str(train_acc) + ' ' + str(val_acc) + ' ' + str(
                test_sigma_acc) + "\n")

    train_acc_list = []
    val_acc_list = []
    test_sigma_acc_list = []

    # train
    for epoch in range(0, args.n_epoch):

        scheduler1.step()
        print('Learning rate: ', optimizer1.state_dict()['param_groups'][0]['lr'])

        train_acc = train(train_loader, epoch, clf1, optimizer1, args, nn.CrossEntropyLoss())
        train_acc_list.append(train_acc)

        val_acc = evaluate(val_loader, clf1)
        val_acc_list.append(val_acc)

        test_sigma_acc = evaluate_test(test_loader, clf1)
        test_sigma_acc_list.append(test_sigma_acc)


        # save results
        print('Epoch [%d/%d] Test Sigma Accuracy on the %s test data: Model1 %.4f %% ' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_sigma_acc))

        with open(txtfile, "a") as myfile:
            myfile.write(
                str(int(epoch)) + ' ' + str(train_acc) + ' ' + str(val_acc) + ' ' + str(test_sigma_acc) + "\n")

        id = np.argmax(np.array(val_acc_list))
        test_sigma_acc_max = test_sigma_acc_list[id]
        print('*********** Best sigma accuracy : %.2f  *********************' % test_sigma_acc_max)

    with open(txtfile, "a") as myfile:
        print('seed:', args.seed)
        myfile.write('*********** Best sigma accuracy : %.2f  *********************\n' % test_sigma_acc_max)

    return test_sigma_acc_max


if __name__ == '__main__':
    best_acc = main(args)

