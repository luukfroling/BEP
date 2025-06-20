import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import optim
import time

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x, return_attention=False):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        if not return_attention:
            return x * psi
        else:
            return x * psi, psi

class AttU_Net(nn.Module):
    def __init__(self,img_ch=10,output_ch=2):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv3d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return torch.sigmoid(d1)  # Apply sigmoid to the output for binary classification
    
    def getAttenuationMap(self, x):
        attention_maps = {}

        # Encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Decoding path with attention maps
        d5 = self.Up5(x5)
        x4, att5 = self.Att5(g=d5, x=x4, return_attention=True)
        attention_maps['att5'] = att5
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3, att4 = self.Att4(g=d4, x=x3, return_attention=True)
        attention_maps['att4'] = att4
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2, att3 = self.Att3(g=d3, x=x2, return_attention=True)
        attention_maps['att3'] = att3
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1, att2 = self.Att2(g=d2, x=x1, return_attention=True)
        attention_maps['att2'] = att2
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        output = torch.sigmoid(d1)

        return output, attention_maps
        


def train_network(network, train_loader, val_loader, device = 'cpu', num_epochs=10, lr=0.1, num_epochs_decay=5):
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(list(network.parameters()), lr=lr)
    start_time = time.time()

    single_mask = torch.zeros((1, 2, 32, 48, 96), dtype=torch.float32)
    single_mask[:, :, :, 8:40, :32] = 1.0  # Set the region of interest to 1
    mask = single_mask.to(device)  # shape = [1, C, 32, 44, 64]

    for epoch in range(num_epochs):
        network.train(True)
        epoch_loss = 0

        # ---------- TRAINING PHASE ----------

        for i, (images, GT) in enumerate(train_loader):
            images = images.to(device)
            GT = GT.to(device)

            SR = network(images)

            #generate loss mask for [:, 8:40, :32], 0 everywhere else
            
            mask_batched = mask.repeat(SR.size(0), 1, 1, 1, 1)  # Repeat for batch size
            per_voxel_loss = criterion(SR, GT)

            masked_loss = per_voxel_loss * mask_batched

            weight = torch.ones_like(GT)
            weight[GT >0.1] = 100  # or another high value depending on your use case

            weighted_loss = (masked_loss * weight).mean()
            epoch_loss += weighted_loss.item()

            network.zero_grad()
            weighted_loss.backward()
            optimizer.step()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_train_loss:.4f} | Time Elapsed: {time.time() - start_time:.2f}s")

        # ---------- VALIDATION PHASE ----------
    
        network.eval()
        val_loss = 0

        with torch.no_grad():
            for images, GT in val_loader:
                images = images.to(device)
                GT = GT.to(device)

                SR = network(images)

                mask_batched = mask.repeat(SR.size(0), 1, 1, 1, 1)  # Repeat for batch size
                per_voxel_loss = criterion(SR, GT)

                masked_loss = per_voxel_loss * mask_batched

                weight = torch.ones_like(GT)
                weight[GT >0.1] = 100  # or another high value depending on your use case

                weighted_loss = (masked_loss * weight).mean()
                val_loss += weighted_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")
