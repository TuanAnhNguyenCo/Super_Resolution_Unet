import torch
import torch.nn as nn
import torchvision

class FirstFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.blockEncoder = nn.Sequential(
            nn.MaxPool2d(2,2),
            ConvBlock(in_channels,out_channels)
        )
    def forward(self, x):
        return self.blockEncoder(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.blockEncoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
            nn.LeakyReLU(),
        )
        self.conv = ConvBlock(in_channels,out_channels)
    def forward(self, x,skip):
        x  = self.blockEncoder(x)
        x = torch.cat([x,skip],dim = 1)
        return self.conv(x)
    
class FinalOutput(nn.Module):
    # if target img was normalized to 0->1 (type = Sigmoid) 
    # => output will pass to sigmoid function
    def __init__(self, in_channels, out_channels,type = None):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1, 1, 0, bias=False),
            nn.Sigmoid() if type is not None else nn.Sequential(),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, features=[64, 128, 256, 512,1024],type = None,LOW_IMG_HEIGHT = 64):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.resize = torchvision.transforms.Resize((LOW_IMG_HEIGHT*4,LOW_IMG_HEIGHT*4),antialias=True)
        self.in_conv1 = FirstFeature(n_channels,features[0])
        self.in_conv2 = ConvBlock(features[0],features[0])
        self.encoder_block1 = EncoderBlock(features[0],features[1])
        self.encoder_block2 = EncoderBlock(features[1],features[2])
        self.encoder_block3 = EncoderBlock(features[2],features[3])
        self.encoder_block4 = EncoderBlock(features[3],features[4])
        
        self.decoder_block1 = DecoderBlock(features[4],features[3])
        self.decoder_block2 = DecoderBlock(features[3],features[2])
        self.decoder_block3 = DecoderBlock(features[2],features[1])
        self.decoder_block4 = DecoderBlock(features[1],features[0])
        
        self.out_conv = FinalOutput(features[0],n_classes,type=type)
    def forward(self, x):
        x = self.resize(x)
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)
        x2 = self.encoder_block1(x1)
        x3 = self.encoder_block2(x2)
        x4 = self.encoder_block3(x3)
        x5 = self.encoder_block4(x4)
        
        x = self.decoder_block1(x5,x4)
        x = self.decoder_block2(x,x3)
        x = self.decoder_block3(x,x2)
        x = self.decoder_block4(x,x1)
        
        output = self.out_conv(x)
        
        return output
    


if __name__ == "__main__":
    img = torch.rand((1,3,64,64))
    model = Unet(type="Sigmoid")
    print(model(img).shape)