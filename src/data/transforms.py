from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_train_transform(image_size: int = 224, augmentation: str = "none"):
    ops = [
        T.Resize(int(image_size * 256 / 224)),
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
    ]
    if augmentation == "classical":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif augmentation != "none":
        raise ValueError(f"Nieznane augmentation={augmentation!r}")
    ops.extend([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return T.Compose(ops)


def make_eval_transform(image_size: int = 224):
    return T.Compose([
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
