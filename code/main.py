from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable
import pre_process as prep
from data_list import ImageList, ImageList_label
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from model import CDCL
from lr_scheduler import InvScheduler

parser = argparse.ArgumentParser(description='Barlow Twins Training')

parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='2048-2048-2048', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/office/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--gamma',
                    type=float,
                    default=0.001,  
                    help='Inv learning rate scheduler parameter, gamma')
parser.add_argument('--decay_rate',
                    type=float,
                    default=0.75, 
                    help='Inv learning rate scheduler parameter, decay rate')
parser.add_argument('--s-dset-path',
                    type=str,
                    default="./data/office/domain_adaptation_images/amazon_list.txt",  
                    help='Source Domain')
parser.add_argument('--t-dset-path',
                    type=str,
                    default="./data/office/domain_adaptation_images/webcam_list.txt",  
                    help='Target Domain')
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='Initial learning rate')                   
parser.add_argument('--test-10crop',
                    type=bool,
                    default=False)
parser.add_argument('--seed',
                    type=int,
                    default=12345)
parser.add_argument('--num-iterations', default=80000, type=int)

# class Office31dataset(Dataset):
#     def __init__(self, src, trg, transform=None):
#         super(Office31dataset, self).__init__()
#         self.src = src
#         self.trg = trg
#         self.transform = transform
#         # self.label2idx = {"back_pack":0, "bike":1, "bike_helmet":2, "bookcase":3, "bottle":4, "calculator":5, "desk_chair":6,
#         #                   "desk_lamp":7, "desktop_computer":8, "file_cabinet":9, "headphones":10, "keyboard":11, "laptop_computer":12, "letter_tray":13,
#         #                   "mobile_phone":14, "monitor":15, "mouse":16, "mug":17, "paper_notebook":18, "pen":19, "phone":20, "printer":21,
#         #                   "projector":22, "punchers":23, "ring_binder":24, "ruler":25, "scissors":26, "speaker":27, "stapler":28, "tape_dispenser":29,
#         #                   "trash_can":30}

#     def __len__(self):
#         return len(self.trg)

#     def __getitem__(self, idx):
#         src_image = self.src[idx][0]
#         trg_image = self.trg[idx][0]
#         trg_label = self.trg[idx][1]
#         # src_image = self.samples[idx][0][0]
#         # trg_image = self.samples[idx][0][1]
#         # src_label = self.samples[idx][1][0]
#         # trg_label = self.label2idx(self.samples[idx][1][1])
#         # print(src_image)

#         # # if self.transform:
#         # #     src_image = self.transform(src_image)
#         # #     trg_image = self.transform(trg_image)
#         # print(1)

#         return [[src_image, trg_image], trg_label]


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):

    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    config = {}
    config["prep"] = {"test_10crop":args.test_10crop, 'params':{"resize_size":256, "crop_size":224}}
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.batch_size}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":args.batch_size}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":4}}
    

    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=True
    
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    source_list = ['.'+i for i in open(data_config["source"]["list_path"]).readlines()]
    target_list = ['.'+i for i in open(data_config["target"]["list_path"]).readlines()]

    dsets["source"] = ImageList(source_list, \
                                transform=prep_dict["source"])
    dset_loaders["source"] = torch.utils.data.DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=args.num_workers, drop_last=True)
    dsets["target"] = ImageList(target_list, \
                                transform=prep_dict["target"])
    dset_loaders["target"] = torch.utils.data.DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=args.num_workers, drop_last=True)
    print("source dataset len:", len(dsets["source"]))
    print("target dataset len:", len(dsets["target"]))

    dsets["target_label"] = ImageList_label(target_list, \
                            transform=prep_dict["target"])
    dset_loaders["target_label"] = torch.utils.data.DataLoader(dsets["target_label"], batch_size=test_bs, \
            shuffle=False, num_workers=args.num_workers, drop_last=False)

    if args.rank == 0:
        # name = args.s_dset_path + "2" + args.t_dset_path
        name = args.s_dset_path.split("/")[-1].split(".")[0] +"2" + args.t_dset_path.split("/")[-1].split(".")[0]
        checkpoint_dir = Path(os.path.join(args.checkpoint_dir, name))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)

    model = CDCL(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ce_criterion = nn.CrossEntropyLoss()
    parameter_list = [{"params": model.parameters(), "lr": 1}]
    group_ratios = [parameter['lr'] for parameter in parameter_list]
    lr_scheduler = InvScheduler(gamma=args.gamma,
                                decay_rate=args.decay_rate,
                                group_ratios=group_ratios,
                                init_lr=args.lr)
    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        checkpoint_name = args.src+"_" +args.trg+'_checkpoint.pth'
        ckpt = torch.load(args.checkpoint_dir / checkpoint_name,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    # train_dataset, val_dataset, test_dataset = office31(
    #                             source_name = args.src,
    #                             target_name = args.trg,
    #                             seed = 4444,
    #                             same_to_diff_class_ratio = 3,
    #                             image_resize = (256, 256),
    #                             group_in_out = True, # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
    #                             framework_conversion = "pytorch",
    #                             office_path = None, #automatically downloads to "~/data"
    #                         )
    # src_data = torchvision.datasets.ImageFolder(root=f"./domain_adaptation_images/{args.src}")
    # trg_data = torchvision.datasets.ImageFolder(root=f"./domain_adaptation_images/{args.trg}")
    
    # transform = transforms.Compose([
    #         transforms.Resize([256, 256]),
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    #         ])
    # print("DOne")

    # train_dataset = Office31dataset(src_data, trg_data)
    # # train_dataset = train_dataset + val_dataset
    # # print("Summation done")
    # # train_dataset = Office31dataset(train_dataset, transform=transform)
    # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # loader = torch.utils.data.DataLoader(train_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
    #                                     pin_memory=True, sampler=sampler)
    # print(len(loader))
    # for step, data in enumerate(loader):
    #     print(data.shape)
    #     break

    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    start_time = time.time()
    for step in tqdm(range(args.num_iterations), total=args.num_iterations):
        lr_scheduler.adjust_learning_rate(optimizer, step)
        optimizer.zero_grad()
        if step % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if step % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda()
        loss = model(inputs_source, inputs_target, labels_source, ce_criterion)
        loss.backward()
        optimizer.step()
        if step % args.print_freq == 0:
            if args.rank == 0:
                stats = dict(step=step,
                                loss=loss.item(),
                                time=int(time.time() - start_time))
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
    if args.rank == 0:
        # save checkpoint
        state = dict(step=step + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
        torch.save(state, checkpoint_dir / checkpoint_name)
    if args.rank == 0:
        # save final model
        model_name = "resnet50_" + str(args.src) + "_" + str(args.trg) + ".pth"
        torch.save(model.module.backbone.state_dict(),
                   checkpoint_dir / model_name)


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == '__main__':
    main()