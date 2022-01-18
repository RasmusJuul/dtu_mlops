"""
LFW dataloading
"""
import argparse
import time
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform):# -> None:
        self.transform = transform
        self.image_list = glob.glob(path_to_folder+"/*/*.jpg")
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index: int):# -> torch.Tensor:
        with Image.open(self.image_list[index]) as img:
            return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='../../data/lfw', type=str) #'../../data/lfw'
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.visualize_batch:
        # TODO: visualize a batch of images
        imgs = iter(dataloader).next()
        imgs = imgs.detach()
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i in range(len(imgs)):
            img = F.to_pil_image(imgs[i])
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show(block=True)
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')

        # 33.966594505310056+-0.9043833985939811 num_worker= 1
        # 16.349738311767577+-0.7371573238589897 num_worker= 2
        # 12.733003377914429+-1.0033153463065823 num_worker= 3
        # 11.168850994110107+-0.3329560736469041 num_worker= 4
        plt.errorbar([1,2,3,4],[33.966594505310056,16.349738311767577,12.733003377914429,11.168850994110107],
                    yerr=[0.9043833985939811,0.7371573238589897,1.0033153463065823,0.3329560736469041])
        plt.xlabel("num_workers")
        plt.ylabel("time[s]")
        plt.title("Time to load 50 batches")
        # plt.savefig("timing.png")
        plt.show()
