import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x
    

class DeConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        super(DeConv, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_planes, in_planes, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.deconv(x)
        return x
    
class Upsampling(nn.Module):

    def __init__(self, in_planes, scale_factor=2, bias=False):
        # TODO bias=True
        super(Upsampling, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class Conv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
# Search space
def Up_Choices_list(in_planes, scale_factor=2):
    module_list = torch.nn.ModuleList()
    if scale_factor==1:
        conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True)
        )
        module_list.append(conv)
    else:
        module_list.append(DeConv(in_planes=in_planes, out_planes=in_planes, kernel_size=3, stride=scale_factor, padding=1, output_padding=1))
    module_list.append(Upsampling(in_planes, scale_factor=scale_factor))
    return module_list

def Conv_Choices_list(in_planes, out_planes):
    module_list = torch.nn.ModuleList()
    module_list.append(Conv(in_planes, out_planes, kernel_size=1, stride=1, padding=0))
    module_list.append(Conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1))
    module_list.append(Conv(in_planes, out_planes, kernel_size=5, stride=1, padding=2))
    module_list.append(Conv(in_planes, out_planes, kernel_size=7, stride=1, padding=3))
    # module_list.append(nn.Identity())        
    return module_list
    
    
class MultiTaskNetOneShot(nn.Module):

    def __init__(self, n_classes=1, c_classes=1, pretrained_encoder=True):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(MultiTaskNetOneShot, self).__init__()

        base = resnet.resnet18(pretrained=pretrained_encoder)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu
        )
        
        self.max_pool = base.maxpool

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        
        
        self.decoder_layers = torch.nn.ModuleList()
        
        self.decoder_layers.append(Up_Choices_list(512, scale_factor=2))
        self.decoder_layers.append(Conv_Choices_list(512, 256))
        self.decoder_layers.append(Conv_Choices_list(256, 256))
        
        self.decoder_layers.append(Up_Choices_list(256, scale_factor=2))
        self.decoder_layers.append(Conv_Choices_list(256, 128))
        self.decoder_layers.append(Conv_Choices_list(128, 128))
        
        self.decoder_layers.append(Up_Choices_list(128, scale_factor=2))
        self.decoder_layers.append(Conv_Choices_list(128, 64))
        self.decoder_layers.append(Conv_Choices_list(64, 64))
        
        self.decoder_layers.append(Up_Choices_list(64, scale_factor=1))
        self.decoder_layers.append(Conv_Choices_list(64, 64))
        self.decoder_layers.append(Conv_Choices_list(64, 64))
        
        self.decoder_layers.append(Up_Choices_list(64, scale_factor=2))
        self.decoder_layers.append(Conv_Choices_list(64, 32))
        self.decoder_layers.append(Conv_Choices_list(32, 32))
        
        self.decoder_layers.append(Up_Choices_list(32, scale_factor=2))
        self.decoder_layers.append(Conv_Choices_list(32, 32))
        self.decoder_layers.append(Conv_Choices_list(32, 32))
        
        # Segmentation
        self.conv_output = nn.Conv2d(32, n_classes, 1, 1, 0)
        
        # Classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512, c_classes)
        feature_dim = 32+32+64+64+128+256+512
        self.fc = nn.Linear(feature_dim, c_classes)

    def forward(self, x, code):
        # Initial block
        e_c1 = self.in_block(x)
        e_max_pool = self.max_pool(e_c1)
        # Encoder blocks
        e1 = self.encoder1(e_max_pool)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Decoder blocks
        d4 = self.decoder_layers[0][code[0]](e4)  # 0:Upsampling
        d4 = self.decoder_layers[1][code[1]](d4)
        d4 = e3 + self.decoder_layers[2][code[2]](d4)
        
        # d3 = e2 + self.decoder3(d4)
        d3 = self.decoder_layers[3][code[3]](d4)  # 3:Upsampling
        d3 = self.decoder_layers[4][code[4]](d3)
        d3 = e2 + self.decoder_layers[5][code[5]](d3)
        
        # d2 = e1 + self.decoder2(d3)
        d2 = self.decoder_layers[6][code[6]](d3) # 6:Upsampling
        d2 = self.decoder_layers[7][code[7]](d2)
        d2 = e1 + self.decoder_layers[8][code[8]](d2)
        
        # d1 = e_max_pool + self.decoder1(d2)
        d1 = self.decoder_layers[9][code[9]](d2)  # 9:Upsampling  (stride: 1)
        d1 = self.decoder_layers[10][code[10]](d1)
        d1 = e_max_pool + self.decoder_layers[11][code[11]](d1)
        
        # d0 = self.decoder0(d1)
        d0 = self.decoder_layers[12][code[12]](d1)  # 12:Upsampling
        d0 = self.decoder_layers[13][code[13]](d0)
        d0 = self.decoder_layers[14][code[14]](d0)
        
        d_final = self.decoder_layers[15][code[15]](d0)  # 15:Upsampling
        d_final = self.decoder_layers[16][code[16]](d_final)
        d_final = self.decoder_layers[17][code[17]](d_final)
        
        y_segmentation = self.conv_output(d_final)
        
        
        # Classification
        x_avg_pool4 = self.avgpool(e4)
        x_flat4 = x_avg_pool4.reshape(x_avg_pool4.size(0), -1)
        '''
        x_avg_pool3 = self.avgpool(e3)
        x_flat3 = x_avg_pool3.reshape(x_avg_pool3.size(0), -1)
        
        x_avg_pool2 = self.avgpool(e2)
        x_flat2 = x_avg_pool2.reshape(x_avg_pool2.size(0), -1)
        
        x_avg_pool1 = self.avgpool(e1)
        x_flat1 = x_avg_pool1.reshape(x_avg_pool1.size(0), -1)
        '''
        x_avg_pool_d4 = self.avgpool(d4)
        x_flat_d4 = x_avg_pool_d4.reshape(x_avg_pool_d4.size(0), -1)
        
        x_avg_pool_d3 = self.avgpool(d3)
        x_flat_d3 = x_avg_pool_d3.reshape(x_avg_pool_d3.size(0), -1)
        
        x_avg_pool_d2 = self.avgpool(d2)
        x_flat_d2 = x_avg_pool_d2.reshape(x_avg_pool_d2.size(0), -1)
        
        x_avg_pool_d1 = self.avgpool(d1)
        x_flat_d1 = x_avg_pool_d1.reshape(x_avg_pool_d1.size(0), -1)
        
        x_avg_pool_d0 = self.avgpool(d0)
        x_flat_d0 = x_avg_pool_d0.reshape(x_avg_pool_d0.size(0), -1)
        
        x_avg_pool_d_final = self.avgpool(d_final)
        x_flat_d_final = x_avg_pool_d_final.reshape(x_avg_pool_d_final.size(0), -1)
        
        x_flat = torch.cat((x_flat4, x_flat_d4, x_flat_d3, x_flat_d2, x_flat_d1, x_flat_d0, x_flat_d_final), dim=1)
        y_classification = self.fc(x_flat)

        return y_classification, y_segmentation