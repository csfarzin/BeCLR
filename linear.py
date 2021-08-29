import argparse
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
sns.set_theme(style="darkgrid")

#from data_aug.transformation import transform
from model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Net(nn.Module):
    def __init__(self, base_model, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        model = Model(base_model=base_model)
        self.f = model.f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1 = 0.0, 0.0
    total_correct_5, total_num, data_bar = 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.long().cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(dim=-1))
                .any(dim=-1).float()).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(dim=-1))
                .any(dim=-1).float()).item()

            data_bar.set_description(
                '{}{}{} {}Epoch:{} [{}/{}] {}Loss:{} {:.4f} {}ACC@1:{} {:.2f}% {}ACC@5:{} {:.2f}%'
                .format(
                    bcolors.OKCYAN,
                    'Train' if is_train else 'Test ',
                    bcolors.ENDC,
                    bcolors.WARNING, bcolors.ENDC,
                    epoch, epochs,
                    bcolors.WARNING, bcolors.ENDC,
                    total_loss / total_num,
                    bcolors.WARNING, bcolors.ENDC,
                    total_correct_1 / total_num * 100,
                    bcolors.WARNING, bcolors.ENDC,
                    total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='The pretrained model path')
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Dir to save csv')
    parser.add_argument(
        '--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--base_model',
                        default='resnet50',
                        help='dataset name',
                        choices=["resnet18", "resnet50"])
    parser.add_argument('-dataset-name', default='imagenet', help='dataset name')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    dataset_name = args.dataset_name
    base_model = args.base_model
    log_dir = args.log_dir
    k_subnets = 100
    
    if dataset_name == 'imagenet':
        # get test and train transformation
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        # data prepare
        root = '~/../DATA/ILSVRC/Data/CLS-LOC'
        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')

        train_data = ImageFolder(traindir, train_transform)
        
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, drop_last=True)

        test_data = ImageFolder(valdir, test_transform)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True)
    
    print(train_data.classes)
    model = Net(base_model, num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
        
    if torch.cuda.device_count() > 1:
        print("We have available", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    save_name_pre = '{}_{}_{}_{}_{}'.format(
        dataset_name, k_subnets, base_model, batch_size, epochs)
    csv_dir = os.path.join(log_dir, '{}_linear_stats.csv'.format(save_name_pre))
    model_dir = os.path.join(log_dir, '{}_linear_model.pth'.format(save_name_pre))
    fig_dir = os.path.join(log_dir, '{}_loss_acc_linear.png'.format(save_name_pre))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(csv_dir, index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, model_dir)
            
    df = pd.read_csv(csv_dir)
    fig, axes = plt.subplots(2, 3, figsize=(20,10))
    axes[0, 0].set_title('Loss/Train')
    axes[0, 1].set_title('acc@1/test')
    axes[0, 2].set_title('acc@1/train')
    axes[1, 0].set_title('Loss/Test')
    axes[1, 1].set_title('acc@5/test')
    axes[1, 2].set_title('acc@5/train')
    sns.lineplot(ax=axes[0, 0], x="epoch", y="train_loss", data=df)
    sns.lineplot(ax=axes[0, 1], x="epoch", y="test_acc@1", data=df)
    sns.lineplot(ax=axes[0, 2], x="epoch", y="train_acc@1", data=df)
    sns.lineplot(ax=axes[1, 0], x="epoch", y="test_loss", data=df)
    sns.lineplot(ax=axes[1, 1], x="epoch", y="test_acc@5", data=df)
    sns.lineplot(ax=axes[1, 2], x="epoch", y="train_acc@5", data=df)
    fig.savefig(fig_dir)
    