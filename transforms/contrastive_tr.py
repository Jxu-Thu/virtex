from torchvision import transforms
from PIL import ImageFilter
import random
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
def Moco_transform(size = 224):
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    return augmentation

def BYOL_transform(size = 224):
    augmentation = transforms.Compose([
            transforms.RandomApply([
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # BYOL
                ], p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))],
                p = 0.2),
            transforms.RandomResizedCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    return augmentation


