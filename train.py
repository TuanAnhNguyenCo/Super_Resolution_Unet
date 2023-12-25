from utils import train,plot_result
from dataloader import ImageData
from torch.utils.data import DataLoader
from model import Unet
from torchvision import transforms
import torch
import os
import random
import numpy as np
from torch import nn

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
if __name__ == "__main__":
    seed_everything(42)
    BATCH_SIZE = 8
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 1),
        transforms.RandomVerticalFlip(p = 1),]
    )
    train_dataset = ImageData('Khoa_LHR_image/train',64,64, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = ImageData('Khoa_LHR_image/val',64,64)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = 'cuda:0'
    model = Unet(type="Sigmoid").to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr =1e-4)
    
    save_model = './UNET'
    os.makedirs(save_model, exist_ok = True)
    EPOCHS = 100
    best_model, metrics = train(
    model, 'best_model', save_model, optimizer, criterion, train_loader, test_loader, EPOCHS, device)
    plot_result(EPOCHS,
    metrics["train_psnr"],
    metrics["valid_psnr"],
    metrics["train_loss"],
    metrics["valid_loss"]
)