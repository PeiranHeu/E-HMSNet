import os
import torch.nn as nn
import torch
from resnet import Backbone_ResNet152_in3
import torch.nn.functional as F
import numpy as np
from toolbox.dual_self_att import CAM_Module

torch.autograd.set_detect_anomaly(True)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.bn(self.fc1(self.max_pool(x)))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y

class CorrelationModule(nn.Module):
    def  __init__(self, all_channel=64):
        super(CorrelationModule, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel,bias = False)
        self.channel = all_channel
        self.fusion = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)

    def forward(self, exemplar, query): # exemplar: middle, query: rgb or T
        fea_size = exemplar.size()[2:]
        all_dim = fea_size[0]*fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim) #N,C,H*W
        query_flat = query.view(-1, self.channel, all_dim)
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  #batchsize x dim x num, N,H*W,C
        exemplar_corr = self.linear_e(exemplar_t) #
        A = torch.bmm(exemplar_corr, query_flat)
        B = F.softmax(torch.transpose(A,1,2),dim=1)
        #当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        exemplar_out = self.fusion(exemplar_att)

        return exemplar_out



import torch.nn.functional as F


class DoubleSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(DoubleSelfAttention, self).__init__()
        self.query_conv = BasicConv2d(in_channels, in_channels // 8, 1)
        self.key_conv = BasicConv2d(in_channels, in_channels // 8, 1)
        self.value_conv = BasicConv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.size = [4, 512, 15, 20]

    def forward(self, x1, x2):
        # x1 = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=False)
        # print(x1.shape)
        # print(x2.shape)
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)
        batch_size, C, width, height = x1.size()
        proj_query = self.query_conv(x1).view(batch_size, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        # 打印张量的形状
        # print(self.key_conv(x2).shape)
        proj_key = self.key_conv(x2).view(batch_size, -1, width * height)  # B X C x (*W*H)

        # print(proj_query.shape)
        # print(proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = F.softmax(energy, dim=-1)  # BX (N) X (N)

        proj_value = self.value_conv(x2).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x1
        # print(out.shape)
        return out


class CLM(nn.Module):
    def __init__(self, all_channel=64):
        super(CLM, self).__init__()
        self.corr_x_2_x_ir = CorrelationModule(all_channel)
        self.corr_ir_2_x_ir = CorrelationModule(all_channel)
        self.smooth1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.smooth2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.fusion = BasicConv2d(2*all_channel, all_channel, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias = True)

    def forward(self, x, x_ir, ir):  # exemplar: middle, query: rgb or T
        corr_x_2_x_ir = self.corr_x_2_x_ir(x_ir,x)
        corr_ir_2_x_ir = self.corr_ir_2_x_ir(x_ir,ir)

        summation = self.smooth1(corr_x_2_x_ir + corr_ir_2_x_ir)
        multiplication = self.smooth2(corr_x_2_x_ir * corr_ir_2_x_ir)

        fusion = self.fusion(torch.cat([summation,multiplication],1))
        sal_pred = self.pred(fusion)

        return fusion, sal_pred


class CAM(nn.Module):
    def __init__(self, all_channel=64):
        super(CAM, self).__init__()
        #self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.sa = SpatialAttention()
        # self-channel attention
        self.cam = SEBlock(all_channel)

    def forward(self, x, ir):
        multiplication = x * ir
        summation = self.conv2(x + ir)

        sa = self.sa(multiplication)
        summation_sa = summation.mul(sa)

        sc_feat = self.cam(summation_sa)

        return sc_feat


class ESM(nn.Module):
    def __init__(self, all_channel=64):
        super(ESM, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(all_channel, all_channel, kernel_size=3,padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias = True)

    def forward(self, x, ir):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(x + ir)
        fusion = (summation + multiplication)
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))
        edge_pred = self.pred(out)

        return out, edge_pred




class prediction_decoder(nn.Module):
    def __init__(self, channel1=64, channel2=128, channel3=256, channel4=256, channel5=512, n_classes=5):
        super(prediction_decoder, self).__init__()
        # 15 20
        self.decoder5 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel5, channel5, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel5, channel4, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 30 40
        self.decoder4 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel4, channel4, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel4, channel3, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 60 80
        self.decoder3 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel3, channel3, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel3, channel2, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 120 160
        self.decoder2 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel2, channel2, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel2, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        self.semantic_pred2 = nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
        # 240 320 -> 480 640
        self.decoder1 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=3, dilation=3),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 480 640
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
                )

    def forward(self, x5, x4, x3, x2, x1):
        # MFNet
        x5_decoder = self.decoder5(x5)
        # for PST900 dataset
        # since the input size is 720x1280, the size of x5_decoder and x4_decoder is 23 and 45, so we cannot use 2x upsampling directrly.
        x5_decoder = F.interpolate(x5_decoder, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x4_decoder = self.decoder4(x5_decoder + x4)
        x3_decoder = self.decoder3(x4_decoder + x3)
        x2_decoder = self.decoder2(x3_decoder + x2)
        semantic_pred2 = self.semantic_pred2(x2_decoder)
        semantic_pred = self.decoder1(x2_decoder + x1)

        return semantic_pred,semantic_pred2

class CF(nn.Module):
    # 特征修正模块
    def __init__(self, input_channel, output_channel):
        super(CF, self).__init__()

        self.max_pool = nn.MaxPool2d(2)
        # self.softconv = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0)
        self.softconv = BasicConv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = BasicConv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.matching_conv = BasicConv2d(output_channel, output_channel, kernel_size=1)

    def forward(self, x):
        x = self.softconv(self.max_pool(x))
        y = x
        x = self.conv1(self.conv2(x))
        x = self.matching_conv(x)
        cf = self.relu1(x + y)
        return cf
        







class LASNet(nn.Module):
    def __init__(self, n_classes):
        super(LASNet, self).__init__()

        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet152_in3(pretrained=True)

        # reduce the channel number, input: 480 640
        self.rgbconv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)   # 240 320
        self.rgbconv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 120 160
        self.rgbconv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 60 80
        self.rgbconv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1) # 30 40
        self.rgbconv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1) # 15 20

        self.CLM5 = CLM(512)
        self.CAM4 = CAM(256)
        self.CAM3 = CAM(256)
        self.CAM2 = CAM(128)
        self.ESM1 = ESM(64)

        self.decoder = prediction_decoder(64,128,256,256,512, n_classes)

        self.CF5 = CF(256, 512)
        self.CF4 = CF(256, 256)
        self.CF3 = CF(128, 256)
        self.CF2 = CF(64, 128)

        self.SAF5 = DoubleSelfAttention(512)
        self.SAF4 = DoubleSelfAttention(256)
        self.SAF3 = DoubleSelfAttention(256)
        self.SAF2 = DoubleSelfAttention(128)

        self.pred = nn.Conv2d(512, 2, kernel_size=3, padding=1, bias=True)
        
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 32
        if imgs.shape[2] % p != 0 or imgs.shape[3] % p != 0 :
            p = 40

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        #print('划分时h', h)
        #print('划分时w', w)
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        #print('分块后xshape1', x.shape[1])
        return x

    def unpatchify(self, x, IMG):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        #print('重组时xshape1', x.shape)
        p = 32
        if IMG.shape[2] == 720:
            p = 40
        #print('重组时P', p)
        h = IMG.shape[2] // p
        w = IMG.shape[3] // p
        #print('重组时h*w', h*w)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs


    def random_masking(self, x1, x2, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x1, x2: [N, L, D], sequences
        """
        N, L, D = x1.shape  # batch, length, dim
        len_keep_x1 = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x1.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # create mask for x1
        mask_x1 = torch.zeros([N, L], dtype=torch.bool, device=x1.device)  # creating a mask tensor for x1

        for i in range(N):
            mask_x1[i, ids_shuffle[i, :len_keep_x1]] = True  # setting True for the indices to keep in x1

        # mask x1
        x1_masked = x1 * mask_x1.unsqueeze(-1)  # apply the mask to x1

        # create mask for x2 based on x1 masked positions
        mask_x2 = ~mask_x1  # create a mask for x2 based on x1's masked positions

        # determine the number of patches needed to mask in x2
        num_patches_to_mask_x2 = int((L - len_keep_x1) * mask_ratio)

        # get additional patches to mask in x2 while avoiding x1 masked positions
        additional_patches = ids_shuffle[:, len_keep_x1:]
        indices_unmasked_x1 = ids_shuffle[:, :len_keep_x1]

        # Create mask for additional patches in x2 while avoiding x1 masked positions
        mask_additional_x2 = torch.ones_like(mask_x2)  # Initialize additional mask for x2

        for i in range(N):
            indices_unmasked_x1_i = indices_unmasked_x1[i]
            additional_patches_i = additional_patches[i]

            for patch in additional_patches_i:
                if patch not in indices_unmasked_x1_i:
                    mask_additional_x2[i, patch] = 0

        # mask x2
        x2_masked = x2 * mask_x2.unsqueeze(-1)  # apply the mask to x2
        x2_masked += x2 * mask_additional_x2.unsqueeze(-1)  # apply additional mask to x2

        return x1_masked, x2_masked, ids_shuffle
        
    def add_gaussian_noise(self, image, mean=0., var=0.00005):
        device = image.device
        noise = torch.randn(image.size()).to(device) * var + mean
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0., 255)  # 将值裁剪到0和255之间
        return noisy_image
  

    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)
        #print('原来x:', x.shape)
        #print('原来ir:', ir.shape)
     

        rgb_patches = self.patchify(x)
        #print('rgb:', rgb_patches.shape)
        t_patches = self.patchify(ir)
        #print('t:', t_patches.shape)
        
        x_m,ir_m,rst = self.random_masking(rgb_patches, t_patches, 0.05)

        x = self.unpatchify(x_m, x)
        ir = self.unpatchify(ir_m, ir)
        #print('x:', x.shape)
        #print('ir:', ir.shape)
        

        #x_n = self.add_gaussian_noise(x)
        #ir_n = self.add_gaussian_noise(ir)

        #x = x_n
        #ir = ir_n

        dif0 = torch.abs(x - ir)
        dif0 = dif0.mean(dim=1, keepdim=True)
        dif0 = torch.sigmoid(dif0)
        # print('dif0:', dif0.shape)



        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)


        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)



        dif5 = torch.abs(x5 - ir5)
        # norm = torch.norm(dif)
        # if norm > 0:
        #     dif /= norm
        dif5 = dif5.mean(dim = 1, keepdim=True)
        dif5 = torch.sigmoid(dif5)
        # print('dif5:', dif5.shape)



        x1 = self.rgbconv1(x1)
        x2 = self.rgbconv2(x2)
        x3 = self.rgbconv3(x3)
        x4 = self.rgbconv4(x4)
        x5 = self.rgbconv5(x5)
        # print('x5', x5.shape)


        ir1 = self.rgbconv1(ir1)
        ir2 = self.rgbconv2(ir2)
        ir3 = self.rgbconv3(ir3)
        ir4 = self.rgbconv4(ir4)
        ir5 = self.rgbconv5(ir5)
        # print('ir5:', ir5.shape)


        #out5, sal = self.CLM5(x5, x5*ir5, ir5)
        # print('经过CLM5后，out5:', out5.shape)
        out4 = self.CAM4(x4, ir4)
        out3 = self.CAM3(x3, ir3)
        out2 = self.CAM2(x2, ir2)


        # old↑
        # new↓
        beta = 0.35

        out5 = dif5 * self.SAF5(self.CF5(x4), ir5) + self.CLM5(dif5*x5, (dif5*x5)*(dif5*ir5), dif5*ir5)[0]
        #out5 = self.SAF5(self.CF5(x4), ir5) + self.CLM5(x5, (x5)*(ir5), ir5)[0]
        # print('经过SAF5后，out5:', out5.shape) + self.SAF5(self.CF5(ir4), x5)
        sal = self.pred(out5)
        # out4 = beta * ( self.SAF4(self.CF4(x3), ir4) + self.SAF4(self.CF4(ir3), x4) ) + self.CAM4(x4, ir4)
        # out3 = beta * self.SAF3(self.CF3(x2), ir3) + self.CAM3(x3, ir3)
        # out2 = dif2 * self.SAF2(self.CF2(dif1 * x1), dif2 * ir2) + dif5 *  self.CAM2(x2, ir2)
        out1, edge = self.ESM1(x1, ir1)



        semantic, semantic2 = self.decoder(out5, out4, out3, out2, out1)
        semantic2 = torch.nn.functional.interpolate(semantic2, scale_factor=2, mode='bilinear')
        sal = torch.nn.functional.interpolate(sal, scale_factor=32, mode='bilinear')
        edge = torch.nn.functional.interpolate(edge, scale_factor=2, mode='bilinear')

        return semantic, semantic2, sal, edge

if __name__ == '__main__':
    torch.cuda.empty_cache()
    # LASNet(9)
    LASNet(8)
    # for PST900 dataset
    # LASNet(5)
