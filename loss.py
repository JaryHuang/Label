# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
from utils.LabelGeneration import LabelGeneration


class CrossEntropyLoss(Module):

    def __init__(self, class_num, reduction="mean", **kwage):
        super(CrossEntropyLoss, self).__init__()
        self.ClassNum = class_num
        self.reduction = reduction

    def getLabelMatrix(self, label):

        # get batch and the number of NL sample
        batch, nlnum = label.size()

        # calculate the one-hot label. Due to the label is [B,NLNUM], so the one-hot need to become the [B*NLNUM,ClassNum]
        one_hot = torch.Tensor([[[0 if i != j else 1 for i in range(self.ClassNum)]
                                 for j in label[k]] for k in range(batch)]).reshape(-1, self.ClassNum).cuda()
        return one_hot, batch, nlnum

    def forward(self, pred, label, regular):
        # get one-hot label
        one_hot, batch, _ = self.getLabelMatrix(label)

        # calculate the log prob
        log_prb = F.log_softmax(pred, dim=1).cuda()

        # calculate loss
        loss = -torch.sum(one_hot * log_prb)
        return loss / batch if self.reduction == 'mean' else loss


class SelNLPL(CrossEntropyLoss):

    def __init__(self, class_num, c, r, negative_num=1, reduction="mean", **kwage):
        super(SelNLPL, self).__init__(class_num)
        self.ClassNum = class_num
        # the SelNL threshold is self.1/C
        self.C = c
        # the SelPL threshold is r
        self.R = r

        self.labelG = LabelGeneration(
            class_num=class_num, negative_num=negative_num)

    def forward(self, pred, label, mode):

        if mode == "NL":
            threshold = 0
        elif mode == "SelNL":
            threshold = 1 / self.C
        elif mode == "SelPL":
            threshold = self.R
        else:
            raise Exception('''the mode:{} is not support'''.format(mode))

        # selected the predict confidence is max than 1/c
        mask, prob, one_hot = self.__SelPrb(pred, label, threshold)

        if torch.sum(mask) == 0:
            loss = -torch.sum(one_hot * torch.log(prob)) * 0.0
            batch = one_hot.size(0)
        elif mode == "NL" or mode == "SelNL":

            nl_label = self.labelG.GetNL(label[mask])
            nl_one_hot, batch, nlnum = self.getLabelMatrix(
                torch.Tensor(nl_label))
            # SelNL calculation
            prb = prob[mask].unsqueeze(1).repeat(
                1, nlnum, 1).reshape(-1, self.ClassNum)

            loss = -torch.sum(nl_one_hot * torch.log(1 - prb))
        else:
            batch = one_hot[mask].size(0)
            loss = -torch.sum(one_hot[mask] * torch.log(prob[mask]))

        #loss = torch.Tensor([5]).cuda().float()*loss

        return loss / batch if self.reduction == 'mean' else loss

    # Select the probably is great than threshold
    def __SelPrb(self, pred, label, threshold):
        prb = F.softmax(pred, dim=1).cuda()
        clamp_prb = torch.clamp(prb, min=1e-8, max=1.0 - 1e-7)
        one_hot, _, _ = self.getLabelMatrix(label)

        mask = clamp_prb[one_hot.byte()] > torch.Tensor([threshold]).cuda()
        return mask, clamp_prb, one_hot


class LabelSmooth(CrossEntropyLoss):

    def __init__(self, alpha, class_num, reduction="mean", **kwage):
        super(LabelSmooth, self).__init__(class_num)
        # labelsmooth parameters
        self.alpha = alpha
        self.ClassNum = class_num
        self.reduction = reduction
    # due to one-hot parameter need change ,so rewrite the funtion. the loss calulation as forward

    def getLabelMatrix(self, label):
        N, M = label.size()
        one_hot = torch.Tensor([[[0 if i != j else 1 for i in range(self.ClassNum)]
                                 for j in label[k]] for k in range(N)]).reshape(-1, self.ClassNum).cuda()
        one_hot = one_hot * (1 - self.alpha) + (1 - one_hot) * \
            self.alpha / (self.ClassNum - 1)
        return one_hot, N, M


if __name__ == "__main__":
    label = torch.Tensor([[0], [3], [4], [1], [2]])
    pred = torch.Tensor([
        [-1.1, -2, -3, 1, 2, 1],
        [2.1, 1, 3, 4, 1, 2],
        [3.2, 4, 2, 2, 2, -3],
        [0.1, 0.2, -0.1, 1, 2, 3],
        [1.1, 1, 2, -3, -5, -1]])

    #loss = CrossEntropyLoss(6)
    #print(loss(pred, label))

    #criterion = nn.CrossEntropyLoss()
    #print(criterion(pred, label.reshape(-1).long()))
    loss = SelNLPL(6, 5, 0.5, negative_num=1)

    print(loss(pred, label, "SelNL"))
