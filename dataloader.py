from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
from PIL import Image


class ImageData(Dataset):
    def __init__(self,img_dir,LOW_IMG_WIDTH, LOW_IMG_HEIGHT,transforms=None):
        super().__init__()
        self.resize = transforms.function.resize((LOW_IMG_WIDTH, LOW_IMG_HEIGHT),antialias=True)
        self.img_dir = img_dir
        self.images = glob.glob(os.path.join(img_dir,"*"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image = Image.open(self.images[idx])
        image = transforms.function.pil_to_tensor(image)
        input_image = (self.resize(image)/255)*2 - 1 # convert to [-1,1]
        target_image = image/255.0
        
        if transforms:
            input_image = transforms(input_image)
            target_image = transforms(target_image)
        return input_image,target_image
        