import torch
import torch.nn as nn
import torchvision
from utils import off_diagonal

class CDCL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        ##### Barlow Twins #####
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [512] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        ##### Supervised learning #####
        # self.classifier = nn.Sequential(nn.Linear(sizes[0], sizes[0]//2, bias=False),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(sizes[0]//2, sizes[0]//4, bias=False),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(sizes[0]//4, sizes[0]//8, bias=False),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(sizes[0]//8, 31, bias=False))
        self.classifier = nn.Sequential(nn.Linear(sizes[0], sizes[0]//2, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(sizes[0]//2, sizes[0]//4, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(sizes[0]//4, sizes[0]//8, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(sizes[0]//8, 31, bias=False))

    def forward(self, src, trg, labels, ce_optim):

        ########################
        ##### Barlow Twins #####
        ########################
        z1 = self.projector(self.backbone(src))
        z2 = self.projector(self.backbone(trg))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        barlow_loss = on_diag + self.args.lambd * off_diag

        ###########################
        ##### Supervised Loss #####
        ###########################
        out = self.classifier(self.backbone(trg))
        ce_loss = ce_optim(out, labels)

        ##### Combined Loss #####
        total_loss = barlow_loss + ce_loss

        return total_loss
