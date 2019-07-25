import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.jigsaw_dataset import JigsawDataset, JigsawTestDataset
from data.rotate_dataset import RotateDataset, RotateTestDataset
from data.concat_dataset import ConcatDataset

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_val_dataloader(args):
    dataset_list = args.name
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    limit = args.limit
    for dname in dataset_list:
        if args.type == 'jigsaw':
            img_transformer, tile_transformer = get_jig_train_transformers(args)
            train_dataset = JigsawDataset(dname, split='train', val_size=args.val_size,
                    img_transformer=img_transformer, tile_transformer=tile_transformer,
                    jig_classes=args.aux_classes, bias_whole_image=args.bias_whole_image)
            val_dataset = JigsawTestDataset(dname, split='val', val_size=args.val_size,
                img_transformer=get_val_transformer(args), jig_classes=args.aux_classes)
        elif args.type == 'rotate':
            img_transformer = get_rot_train_transformers(args)
            train_dataset = RotateDataset(dname, split='train', val_size=args.val_size,
                    img_transformer=img_transformer, rot_classes=args.aux_classes, bias_whole_image=args.bias_whole_image)
            val_dataset = RotateTestDataset(dname, split='val', val_size=args.val_size,
                img_transformer=get_val_transformer(args), rot_classes=args.aux_classes)

        if limit:
            train_dataset = Subset(train_dataset, limit)

        datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader

def get_target_dataloader(args):

    name = args.name
    if args.type == 'jigsaw':
        img_transformer, tile_transformer = get_jig_train_transformers(args)
        dataset = JigsawDataset(name, 'train', img_transformer=img_transformer,
                tile_transformer=tile_transformer, jig_classes=args.aux_classes,
                bias_whole_image=args.bias_whole_image)
    elif args.type == 'rotate':
        img_transformer = get_rot_train_transformers(args)
        dataset = RotateDataset(name, 'train', img_transformer=img_transformer,
                rot_classes=args.aux_classes, bias_whole_image=args.bias_whole_image)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader

def get_test_dataloader(args):
    img_tr = get_val_transformer(args)
    name = args.name
    if args.type == 'jigsaw':
        val_dataset = JigsawTestDataset(name, split='test',
                img_transformer=img_tr, jig_classes=args.aux_classes)
    elif args.type == 'rotate':
        val_dataset = RotateTestDataset(name, split='test',
                img_transformer=img_tr, rot_classes=args.aux_classes)

    if args.limit and len(val_dataset) > args.limit:
        val_dataset = Subset(val_dataset, args.limit)
        print("Using %d subset of dataset" % args.limit)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader

def get_jig_train_transformers(args):
    size = args.img_transform.random_resize_crop.size
    scale = args.img_transform.random_resize_crop.scale
    img_tr = [transforms.RandomResizedCrop((int(size[0]), int(size[1])), (scale[0], scale[1]))]
    if args.img_transform.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.img_transform.random_horiz_flip))
    if args.img_transform.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(
            brightness=args.img_transform.jitter, contrast=args.img_transform.jitter,
            saturation=args.jitter, hue=min(0.5, args.jitter)))

    tile_tr = []
    if args.jig_transform.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.jig_transform.tile_random_grayscale))
    mean = args.normalize.mean
    std = args.normalize.std
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)

def get_rot_train_transformers(args):
    size = args.img_transform.random_resize_crop.size
    scale = args.img_transform.random_resize_crop.scale
    img_tr = [transforms.RandomResizedCrop((int(size[0]), int(size[1])), (scale[0], scale[1]))]
    if args.img_transform.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.img_transform.random_horiz_flip))
    if args.img_transform.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(
            brightness=args.img_transform.jitter, contrast=args.img_transform.jitter,
            saturation=args.jitter, hue=min(0.5, args.jitter)))

    mean = args.normalize.mean
    std = args.normalize.std
    img_tr += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    return transforms.Compose(img_tr)

def get_val_transformer(args):
    size = args.img_transform.random_resize_crop.size
    mean = args.normalize.mean
    std = args.normalize.std
    img_tr = [transforms.Resize(tuple(size)), transforms.ToTensor(),
              transforms.Normalize(mean, std=std)]
    return transforms.Compose(img_tr)
