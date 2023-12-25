from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
from PIL import Image
import random


class ImageData(Dataset):
    def __init__(self,img_dir,LOW_IMG_WIDTH, LOW_IMG_HEIGHT,transform=None):
        super().__init__()
        self.resize = transforms.Resize((LOW_IMG_WIDTH, LOW_IMG_HEIGHT),antialias=True)
        self.img_dir = img_dir
        self.images = glob.glob(os.path.join(img_dir,"*"))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = transforms.functional.pil_to_tensor(image)
        input_image = (self.resize(image)/255)*2 - 1 # convert to [-1,1]
        target_image = image/255.0
        
        if self.transform:
            if random.random() > 0.5 :
                input_image = self.transform(input_image)
                target_image = self.transform(target_image)
        return input_image,target_image
        