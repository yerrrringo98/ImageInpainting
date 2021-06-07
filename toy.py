import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from model.networks import *

# SOLS = []
# SOL1 = torch.zeros((16,16))
# SOL1[0::2, :] = 1
# SOLS.append(SOL1)
# SOL2 = torch.zeros((16,16))
# SOL2[1::2, :] = 1
# SOLS.append(SOL2)
# SOL3 = torch.zeros((16,16))
# SOL3[:, 0::2] = 1
# SOLS.append(SOL3)
# SOL4 = torch.zeros((16,16))
# SOL4[:, 1::2] = 1
# SOLS.append(SOL4)
# SOL5 = torch.zeros((16,16))
# SOL5[0::2, 0::2] = 1
# SOL5[1::2, 1::2] = 1
# SOLS.append(SOL5)
# SOL6 = torch.zeros((16,16))
# SOL6[1::2, 0::2] = 1
# SOL6[0::2, 1::2] = 1
# SOLS.append(SOL6)
# SOL7 = torch.ones((16,16))*0.5
# SOLS.append(SOL7)
# input = torch.ones((8,8))
data_test = []
data_train = []
outputs = []
# for i in range(10):
#     rgb_range = torch.randint(0, 128, (3,))
#     pick_range = torch.randint(0, 7, (1,))
#     LR = rgb_range.view(3,1,1).repeat(1, 8, 8).float()
#     HR = SOLS[pick_range].repeat(3,1,1)*rgb_range.view(3,1,1).float()*2
#     data_test.append((LR, HR, pick_range))

# for i in range(200000):
#     rgb_range = torch.randint(0, 128, (3,))
#     pick_range = torch.randint(0, 7, (1,))
#     LR = rgb_range.view(3,1,1).repeat(1, 8, 8).float()
#     HR = SOLS[pick_range].repeat(3,1,1)*rgb_range.view(3,1,1).float()*2
#     data_train.append((LR, HR, pick_range))
#     print(i)
#
# torch.save(data_train, './data_toy.pth')
# print("successfully saved")
data_train = torch.load('./data_toy.pth')
training_data_loader = DataLoader(dataset=data_train, num_workers=1, batch_size=8, shuffle=True)
testing_data_loader = DataLoader(dataset=data_test, num_workers=1, batch_size=1,
                                 shuffle=False)
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.5, 0.5, 0.5), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.NH = 14
        self.MH = nn.Parameter(torch.empty(self.NH, 1, 16, 1, 1))
        torch.nn.init.zeros_(self.MH)
        self.inconv = nn.Conv2d(3, 16, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv2 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
        self.conv3_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.outconv = nn.Conv2d(16, 3, 1, 1, 0)
        self.add_mean = MeanShift(255, sign=1)
    def forward(self, x):
        out = F.relu(self.inconv(x))
        out = F.relu(self.conv1_1(out))
        out = F.relu(self.conv1_2(out))
        out = F.relu(self.conv1_3(out))
        N, C, H, W = out.size()
        out.unsqueeze(0)
        out = out.repeat(self.NH, 1, 1, 1, 1)
        out = out * torch.sigmoid(self.MH)
        out = out.view(N*self.NH, C, H, W)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = (self.outconv(out))
        out = self.add_mean(out)
        # out = F.sigmoid(out)
        return out


# toynetG =
# toynetD =
toynet = NET()
toynet.cuda()
optimizer = optim.Adam(toynet.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
toynet.train()
# HR = torch.cat([SOL1, SOL2, SOL3, SOL4, SOL5, SOL6], dim=0)
for epoch in range(10):
    print("Epoch %d\n" % epoch)
    for i, (lr, hr, pick) in enumerate(training_data_loader):
        lr, hr = lr.cuda(), hr.cuda()
        HRs = hr.repeat(14, 1, 1, 1, 1)
        optimizer.zero_grad()
        SR = NET().cuda().forward(lr)
        loss = F.mse_loss(SR, HRs.view(-1, 3, 16, 16), reduce=False)
        N, C, H, W = loss.size()
        loss = loss.view(14, N // 14, -1).mean(2)
        # L_cos = abs(F.cosine_similarity(toynet.MH, toynet.MH[1].squeeze(), dim=0, eps=1e-6)) + \
        #         abs(F.cosine_similarity(toynet.MH[1].squeeze(), toynet.MH[2].squeeze(), dim=0, eps=1e-6)) + \
        #         abs(F.cosine_similarity(toynet.MH[2].squeeze(), toynet.MH[3].squeeze(), dim=0, eps=1e-6)) + \
        #         abs(F.cosine_similarity(toynet.MH[3].squeeze(), toynet.MH[4].squeeze(), dim=0, eps=1e-6)) + \
        #         abs(F.cosine_similarity(toynet.MH[4].squeeze(), toynet.MH[5].squeeze(), dim=0, eps=1e-6)) + \
        #         abs(F.cosine_similarity(toynet.MH[5].squeeze(), toynet.MH[6].squeeze(), dim=0, eps=1e-6))
        # loss_mse, idx = torch.min(loss, dim=0)
        # loss = loss_mse.mean()# + abs(L_cos)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if i % 5000 == 0:
            print("Iteration [%d/%d], Loss_MSE: %.5f" % (i, len(training_data_loader), loss))
    with torch.no_grad():
        for i, (lr, hr, pick) in enumerate(testing_data_loader):
            lr, hr = lr.cuda(), hr.cuda()
            SR = toynet(lr)
            path = os.path.join('Toy_result_mean', str(epoch))
            if not os.path.exists(path):
                os.mkdir(path)
            save_image(lr.squeeze()/255, os.path.join(path, 'Test_%d_LR.png') % i)
            save_image(SR[0]/255, os.path.join(path, 'Test_%d_SR_0.png') % i)
            save_image(SR[1]/255, os.path.join(path, 'Test_%d_SR_1.png') % i)
            save_image(SR[2]/255, os.path.join(path, 'Test_%d_SR_2.png') % i)
            save_image(SR[3]/255, os.path.join(path, 'Test_%d_SR_3.png') % i)
            save_image(SR[4]/255, os.path.join(path, 'Test_%d_SR_4.png') % i)
            save_image(SR[5]/255, os.path.join(path, 'Test_%d_SR_5.png') % i)
            save_image(SR[6]/255, os.path.join(path, 'Test_%d_SR_6.png') % i)
            save_image(SR[7]/255, os.path.join(path, 'Test_%d_SR_7.png') % i)
            save_image(SR[8]/255, os.path.join(path, 'Test_%d_SR_8.png') % i)
            save_image(SR[9]/255, os.path.join(path, 'Test_%d_SR_9.png') % i)
            save_image(SR[10]/255, os.path.join(path, 'Test_%d_SR_10.png') % i)
            save_image(SR[11]/255, os.path.join(path, 'Test_%d_SR_11.png') % i)
            save_image(SR[12]/255, os.path.join(path, 'Test_%d_SR_12.png') % i)
            save_image(SR[13]/255, os.path.join(path, 'Test_%d_SR_13.png') % i)
            save_image(hr.squeeze()/255, os.path.join(path, 'Test_%d_HR.png') % i)
# print("SOL1", SOL1)
# print("SOL2", SOL2)
# print("SOL3", SOL3)
# print("SOL4", SOL4)
# print("SOL5", SOL5)
# print("SOL6", SOL6)
# print("SOL7", SOL7)
# print("MH1", toynet.MH)
# print("MH2", toynet.MH[1].squeeze())
# print("MH3", toynet.MH[2].squeeze())
# print("MH4", toynet.MH[3].squeeze())
# print("MH5", toynet.MH[4].squeeze())
# print("MH6", toynet.MH[5].squeeze())
# print("MH7", toynet.MH[6].squeeze())
# Iteration [0/25000], Loss_MSE: 11055.19727
# Iteration [5000/25000], Loss_MSE: 5424.17383
# Iteration [10000/25000], Loss_MSE: 6247.83203
# Iteration [15000/25000], Loss_MSE: 4333.69727
# Iteration [20000/25000], Loss_MSE: 2632.96997
# Epoch 1
#
# Iteration [0/25000], Loss_MSE: 2353.30737
# Iteration [5000/25000], Loss_MSE: 1370.04919
# Iteration [10000/25000], Loss_MSE: 710.81427
# Iteration [15000/25000], Loss_MSE: 346.69501
# Iteration [20000/25000], Loss_MSE: 167.17815
# Epoch 2
#
# Iteration [0/25000], Loss_MSE: 116.83627
# Iteration [5000/25000], Loss_MSE: 63.60677
# Iteration [10000/25000], Loss_MSE: 65.25604
# Iteration [15000/25000], Loss_MSE: 55.66267
# Iteration [20000/25000], Loss_MSE: 38.30841
# Epoch 3
#
# Iteration [0/25000], Loss_MSE: 60.99349
# Iteration [5000/25000], Loss_MSE: 33.25347
# Iteration [10000/25000], Loss_MSE: 30.74792
# Iteration [15000/25000], Loss_MSE: 20.40086
# Iteration [20000/25000], Loss_MSE: 20.76929
# Epoch 4
#
# Iteration [0/25000], Loss_MSE: 23.85748
# Iteration [5000/25000], Loss_MSE: 28.59535
# Iteration [10000/25000], Loss_MSE: 32.34833
# Iteration [15000/25000], Loss_MSE: 33.25499
# Iteration [20000/25000], Loss_MSE: 27.33622
# Epoch 5
#
# Iteration [0/25000], Loss_MSE: 19.85824
# Iteration [5000/25000], Loss_MSE: 15.93676
# Iteration [10000/25000], Loss_MSE: 10.42387
# Iteration [15000/25000], Loss_MSE: 18.02364
# Iteration [20000/25000], Loss_MSE: 16.57135
# Epoch 6
#
# Iteration [0/25000], Loss_MSE: 7.19311
# Iteration [5000/25000], Loss_MSE: 22.53238
# Iteration [10000/25000], Loss_MSE: 13.35308
# Iteration [15000/25000], Loss_MSE: 8.91064
# Iteration [20000/25000], Loss_MSE: 17.29352
# Epoch 7
#
# Iteration [0/25000], Loss_MSE: 26.34810
# Iteration [5000/25000], Loss_MSE: 9.20024
# Iteration [10000/25000], Loss_MSE: 10.78049
# Iteration [15000/25000], Loss_MSE: 13.55290
# Iteration [20000/25000], Loss_MSE: 14.53043
# Epoch 8
#
# Iteration [0/25000], Loss_MSE: 14.61833
# Iteration [5000/25000], Loss_MSE: 12.32517
# Iteration [10000/25000], Loss_MSE: 6.45882
# Iteration [15000/25000], Loss_MSE: 11.53392
# Iteration [20000/25000], Loss_MSE: 18.36792
# Iteration [0/25000], Loss_MSE: 10680.21484
# Iteration [5000/25000], Loss_MSE: 5467.89209
# Iteration [10000/25000], Loss_MSE: 4725.74170
# Iteration [15000/25000], Loss_MSE: 3579.03809
# Iteration [20000/25000], Loss_MSE: 3967.62207
# Epoch 1
#
# Iteration [0/25000], Loss_MSE: 3906.35962
# Iteration [5000/25000], Loss_MSE: 4244.63525
# Iteration [10000/25000], Loss_MSE: 5839.10693
# Iteration [15000/25000], Loss_MSE: 3742.62817
# Iteration [20000/25000], Loss_MSE: 5227.83057
# Epoch 2
#
# Iteration [0/25000], Loss_MSE: 4286.83594
# Iteration [5000/25000], Loss_MSE: 5887.56055
# Iteration [10000/25000], Loss_MSE: 3710.49243
# Iteration [15000/25000], Loss_MSE: 4823.10107
# Iteration [20000/25000], Loss_MSE: 4267.29297
# Epoch 3
#
# Iteration [0/25000], Loss_MSE: 4609.49561
# Iteration [5000/25000], Loss_MSE: 3839.43311
# Iteration [10000/25000], Loss_MSE: 5586.51709
# Iteration [15000/25000], Loss_MSE: 3316.60669
# Iteration [20000/25000], Loss_MSE: 4873.22998
# Epoch 4
#
# Iteration [0/25000], Loss_MSE: 6154.97461
# Iteration [5000/25000], Loss_MSE: 4843.81738
# Iteration [10000/25000], Loss_MSE: 6195.00928
# Iteration [15000/25000], Loss_MSE: 4191.90918
# Iteration [20000/25000], Loss_MSE: 3661.20166
# Epoch 5
#
# Iteration [0/25000], Loss_MSE: 3026.22803
# Iteration [5000/25000], Loss_MSE: 4133.22461
# Iteration [10000/25000], Loss_MSE: 4587.63428
# Iteration [15000/25000], Loss_MSE: 6779.46777
# Iteration [20000/25000], Loss_MSE: 5585.77246
# Epoch 6
#
# Iteration [0/25000], Loss_MSE: 4060.62524
# Iteration [5000/25000], Loss_MSE: 4363.52490
# Iteration [10000/25000], Loss_MSE: 4807.26807
# Iteration [15000/25000], Loss_MSE: 2865.36743
# Iteration [20000/25000], Loss_MSE: 4030.68652
# Epoch 7
#
# Iteration [0/25000], Loss_MSE: 4234.62012
# Iteration [5000/25000], Loss_MSE: 6232.44971
# Iteration [10000/25000], Loss_MSE: 5440.59473
# Iteration [15000/25000], Loss_MSE: 5087.11133
# Iteration [20000/25000], Loss_MSE: 5974.15771
# Epoch 8
#
# Iteration [0/25000], Loss_MSE: 4468.36523
# Iteration [5000/25000], Loss_MSE: 5480.41992
# Iteration [10000/25000], Loss_MSE: 4586.82275
# Iteration [15000/25000], Loss_MSE: 3932.19214
# Iteration [20000/25000], Loss_MSE: 4161.69043
# Epoch 9
#
# Iteration [0/25000], Loss_MSE: 5406.06152
# Iteration [5000/25000], Loss_MSE: 5323.65918
# Iteration [10000/25000], Loss_MSE: 7101.52881
# Iteration [15000/25000], Loss_MSE: 5797.65088
# Iteration [20000/25000], Loss_MSE: 7295.40771
#
# Process finished with exit code 0