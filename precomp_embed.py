import torch
import torchvision
import numpy as np
import cv2
import os
join = os.path.join
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from skimage import transform, io, segmentation
import monai
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# save_folder = "sam_train_demo/train/embeds/"
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)

def reshape(np_arr):
    arr_shape = np_arr.shape
    return np_arr.reshape((arr_shape[2], arr_shape[0], arr_shape[1]))

def reshape(np_arr):
    arr_shape = np_arr.shape
    return np_arr.reshape((arr_shape[2], arr_shape[0], arr_shape[1]))


class SAMTransferDataset(Dataset):
    def __init__(self, root, split="train", transform=None, transform_mask=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.transform_mask = transform_mask
        self.img_size = 256 # Default 256
        self.label_id = 255

        # Load the list of image filenames
        self.filenames = []
        with open(os.path.join(self.root, "metadata/{}.txt".format(split)), "r") as f:
            for line in f:
                self.filenames.append(line.strip())
    
    def get_split(self):
        return self.split
    
    def get_root(self):
        return self.root
                
    def reshape(self, np_arr):
        arr_shape = np_arr.shape
        return np_arr.reshape((arr_shape[2], arr_shape[0], arr_shape[1]))

    def __getitem__(self, index):
        # Load the image and segmentation mask
        filename = self.filenames[index]
        image_path = self.split + "/images/{}.png".format(filename)
        mask_path = self.split + "/labels/{}.png".format(filename)
        image = Image.open(os.path.join(self.root, image_path))
        base_mask = Image.open(os.path.join(self.root, mask_path))
        
        # label
        label = np.array(base_mask)
        label = transform.resize(
            label == self.label_id,
            (self.img_size, self.img_size),
            order=0,
            preserve_range=True,
            mode="constant"
        )
        label = np.uint8(label)
        
        # Pre embed prep
        image_pre_emb = np.array(image)
        image_pre_emb = image_pre_emb[:,:,:3]
        image_pre_emb = transform.resize(
            image_pre_emb,
            (self.img_size, self.img_size),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )
        image_pre_emb = np.uint8(image_pre_emb)
        return image_pre_emb, label, filename
    
    def __len__(self):
        return len(self.filenames)
    
def compute_and_save_embeds(dataset, model):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    split = dataset.get_split()
    root = dataset.get_root()
    device = next(model.parameters()).device.type
    
    for i, (observations, labels, filename) in enumerate(loader):
        save_path = root + "/" + split + "/embeds/{}.npy".format(filename[0])

        embed_prepped_obs = observations[0,:,:,:]
        embed_prepped_obs = reshape(embed_prepped_obs)

        # Data to device
        observations = observations.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(axis=0)

        # Embedding prep
        sam_transform = ResizeLongestSide(model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(embed_prepped_obs)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        input_image = model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)

        #Embed
        image_embeddings = None
        with torch.no_grad():
            # Img Embed
            image_embeddings = model.image_encoder(input_image)
            image_embeddings = image_embeddings.squeeze().cpu().numpy()
            np.save(save_path, image_embeddings)
    
    
# Load the training and validation datasets
train_dataset = SAMTransferDataset(root="sam_train_demo", split="train", transform=None, transform_mask=None)
#val_dataset = SAMTransferDataset(root="sam_train_demo", split="val", transform=None, transform_mask=None)

model_type = 'vit_h'
checkpoint = 'sam_vit_h_4b8939.pth'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(DEVICE)

compute_and_save_embeds(train_dataset, sam_model)
# compute_and_save_embeds("val", root="sam_train_demo")



        
    