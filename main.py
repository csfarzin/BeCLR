import argparse
import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


from data_aug.data_loader import CustomDataLoader
from model import Model
from trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--fc_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k_nn', default=200, type=int, help='k in knn')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--epochs', default=500, type=int, help='epochs')
    parser.add_argument('--k_subs', default=100, type=int, help='k subnets')
    parser.add_argument('--layer_size', default=[128, 1], type=int,
                        help='subnetworks layers size (defaut: [64, 1])')
    parser.add_argument('--lr', default=5e-3, type=float,help='initial learning rate')
    parser.add_argument('--wd', default=1e-6, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', default=10, type=int, help='seed for initializing training.')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--base_model',
                        default='resnet50',
                        help='dataset name',
                        choices=["resnet18", "resnet50"])
    
    parser.add_argument('-dataset-name', default='cifar100',
                    help='dataset name', choices=['stl10',
                                                  'cifar10',
                                                  'cifar100',
                                                  'brain',
                                                  'skin',
                                                  'retina',
                                                  'iris',
                                                  'imagenet'
                                                  'coco'])
    # args parse
    args = parser.parse_args()
    base_model = args.base_model
    dataset_name = args.dataset_name
    lr, wd = args.lr, args.wd
    fc_dim, temperature, k_nn = args.fc_dim, args.temperature, args.k_nn
    batch_size, epochs = args.batch_size, args.epochs
    workers = args.workers

    # model setup and optimizer config
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        #cudnn.deterministic = False
        #cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        
    # create a tensorboard writer
    writer = SummaryWriter()
    # save config file
    save_config_file(writer.log_dir, args)
############################################################]
### Load Datasets and Dataloaders
    dl = CustomDataLoader()
    train_loader, memory_loader, test_loader = dl.get_loader(dataset_name, batch_size, workers)
    
    num_cls = len(test_loader.dataset.classes)
    model = Model(base_model=base_model,
                  fc_dim=fc_dim,
                  k_subs=args.k_subs,
                  layer_sizes=args.layer_size,
                  use_bn=False,
                  dr_rate=0.2).to(args.device)
    print(model)
    if torch.cuda.device_count() > 1:
            print("We have available", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model, device_ids=[0,1])
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=0, 
    last_epoch=-1)
    
    trainer = Trainer(model,
                      optimizer,
                      scheduler,
                      temperature,
                      num_cls,
                      epochs,
                      args.device)

    # training loop
    results = {'train_loss': [],
               'bloss_loss': [],
               'NTXent_loss': [],
               'test_acc@1': [],
               'test_acc@5': []
              }
    save_name_pre = '{}_K{}_{}_{}_{}_{}_{}_{}_{}'.format(
        dataset_name, args.k_subs,
        base_model, lr,
        fc_dim, temperature,
        k_nn, batch_size, epochs)
    csv_dir = os.path.join(writer.log_dir, '{}_stats.csv'.format(save_name_pre))
    model_dir = os.path.join(writer.log_dir, '{}_model.pth'.format(save_name_pre))
    fig_dir = os.path.join(writer.log_dir, '{}_loss_acc.png'.format(save_name_pre))
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, bloss, NTXent = trainer.train(train_loader, epoch)
        results['train_loss'].append(train_loss)
        results['bloss_loss'].append(bloss)
        results['NTXent_loss'].append(NTXent)
        writer.add_scalar('loss/train', results['train_loss'][-1], epoch)
        
        test_acc_1, test_acc_5 = trainer.test(memory_loader, test_loader, k_nn, epoch)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        writer.add_scalar('acc@1/test', results['test_acc@1'][-1], epoch)
        writer.add_scalar('acc@5/test', results['test_acc@5'][-1], epoch)
        
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
    
    # plotting loss and accuracies
    df = pd.read_csv(csv_dir)
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(20,5))
    axes[0].set_title('Loss/Train')
    axes[1].set_title('acc@1/test')
    axes[2].set_title('acc@5/test')
    sns.lineplot(ax=axes[0], x="epoch", y="train_loss", data=df)
    sns.lineplot(ax=axes[1], x="epoch", y="test_acc@1", data=df)
    sns.lineplot(ax=axes[2], x="epoch", y="test_acc@5", data=df)
    
    fig.savefig(fig_dir)
    
    
