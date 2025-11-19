import albumentations as A
from albumentations.pytorch import ToTensorV2
from .transforms import Compose, LoadImage, LoadMask, PhotometricDistortion, AlbumentationsTransform, RandomErasing, RandomScale, Pad, ToTensor,  RandomCrop, Resize, Normalize

def get_train_transform(target_size=(512,512)):
    
    pipeline = Compose([
                LoadImage(),
                LoadMask(), 
                Resize((target_size[0], target_size[1])),
                AlbumentationsTransform(A.HorizontalFlip(p=.5)),  # Albumentations
                PhotometricDistortion(),
                RandomScale((0.7, 1.5)),
                RandomCrop((target_size[0], target_size[1])),
                Pad((target_size[0], target_size[1])),
                RandomErasing(prob=0.5),

                Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
                ToTensor()
                ])
    
    return pipeline

