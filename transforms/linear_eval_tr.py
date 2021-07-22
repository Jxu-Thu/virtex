from torchvision import transforms
import torchvision.transforms.functional as F

class SmallestMaxSize(transforms.Resize):
    def __init__(self, max_size = 256):
        super().__init__(max_size)
        self.max_size = max_size
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        width, height = img.size
        min_edge = min(width, height)
        scale = self.max_size/min_edge
        height, width = height*scale, width*scale
        return F.resize(img, (height, width), self.interpolation)


def resize_transform(size = 224):
    augmentation = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    return augmentation

def train_transform(size = 224):
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    return augmentation

def val_transform(size = 224):
    augmentation = transforms.Compose([
        SmallestMaxSize(),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    return augmentation