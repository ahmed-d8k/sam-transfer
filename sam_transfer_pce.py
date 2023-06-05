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

save_folder = "output/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
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
                
    def reshape(self, np_arr):
        arr_shape = np_arr.shape
        return np_arr.reshape((arr_shape[2], arr_shape[0], arr_shape[1]))

    def __getitem__(self, index):
        # Load the image and segmentation mask
        filename = self.filenames[index]
        # image_path = self.split + "/images/{}.png".format(filename)
        mask_path = self.split + "/labels/{}.png".format(filename)
        image_embedding_path = self.split + "/embeds/{}.npy".format(filename)
        # image = Image.open(os.path.join(self.root, image_path))
        base_mask = Image.open(os.path.join(self.root, mask_path))
        image_embedding = np.load(os.path.join(self.root, image_embedding_path))
        
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
        
        
        # Convert to tensors
        return image_embedding, label
    
    def __len__(self):
        return len(self.filenames)
    
# Load the training and validation datasets
train_dataset = SAMTransferDataset(root="sam_train_demo", split="train", transform=None, transform_mask=None)
#val_dataset = SAMTransferDataset(root="sam_train_demo", split="val", transform=None, transform_mask=None)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)

model_type = 'vit_h'
checkpoint = 'sam_vit_h_4b8939.pth'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(DEVICE)
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0) #1e-5 default
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
scheduler = MultiStepLR(optimizer, milestones=[70, 110, 140], gamma=0.1)

model_pckg = (sam_model, seg_loss, optimizer, scheduler)

def train_model(model_package,
                train_loader,
                val_loader,
                num_epochs=1,
                early_stop_criterion=10,
                report_modifier=0.05,
                record_modifier=0.05,
                model_save_name="model.pth",
                save_model=False):
    print("Starting model training...")

    # Important Variable Setup
    model = model_package[0]
    criterion = model_package[1]
    optimizer = model_package[2]
    scheduler = model_package[3]
    device = next(model.parameters()).device.type
    train_losses = []
    train_loss = 0
    val_losses = []
    val_loss = 0
    best_val_loss = 0
    early_stop_test_num = 0
    early_stop_criterion_met = False
    first = True
    n_total_steps = len(train_loader)
    report_freq = int(n_total_steps*report_modifier)
    if report_freq == 0:
        report_freq = 1
    record_freq = int(n_total_steps*record_modifier)
    if record_freq == 0:
        record_freq = 1

    # Train Loop
    for epoch in range(num_epochs):
        if early_stop_criterion_met:
            break
        for i, (image_embeddings, labels) in enumerate(train_loader):
            if early_stop_criterion_met:
                break
            # This may be unnecesarily set but is here to avoid accidental
            # bugs where it is not set
            model.train()
            
            # Data to device
            labels = labels.to(device)
            labels = labels.unsqueeze(axis=0)
            image_embeddings = image_embeddings.to(device)
            
            
            
            sparse_embeddings = None
            dense_embeddings = None
            with torch.no_grad():
                # set the bbox as the image size for fully automatic segmentation 
                B, _, H, W = labels.shape
                boxes = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().to(device)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=boxes[:, None, :],
                    masks=None,
                )

            # Forward pass
            outputs, _ = model.mask_decoder(
                image_embeddings=image_embeddings, # (B, 256, 64, 64)
                image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64) # ATM this outputs (B, 1, 256, 256)
                multimask_output=False,
            )
            
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss Accumulation
            train_loss += loss.item()

            # Report Progress
            if (i+1) % report_freq == 0:
                progress_str = f'Epoch [{epoch+1}/{num_epochs}], '
                progress_str += f'Step [{i+1}/{n_total_steps}],'
                progress_str += f'Loss: {loss.item():.4f}'
                print(progress_str)

            # Pass through validation set and record Val Loss and
            # Accumulated Train Loss Reset both when done
            if (i+1) % record_freq == 0:
                train_loss /= record_freq
                train_losses.append(train_loss)
                train_loss = 0
                if val_loader is not None:
                    val_loss = 0
                    for i, (observations, labels) in enumerate(val_loader):
                        observations = observations.to(device)
                        labels = labels.to(device)
                        model.eval()
                        with torch.inference_mode():
                            outputs = model(observations)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()

                    val_loss /= len(val_loader)
                    val_losses.append(val_loss)

                    if first:
                        best_val_loss = val_loss
                        first = False
                    elif best_val_loss >= val_loss:
                        early_stop_test_num = 0
                        best_val_loss = val_loss
                    elif best_val_loss < val_loss:
                        early_stop_test_num += 1
                        if early_stop_test_num == early_stop_criterion:
                            print("Training stopping early...")
                            early_stop_criterion_met = True
        scheduler.step()
    if save_model:
        print('Finished Training, Saving model...')
        model_save_path = save_folder + model_save_name
        torch.save(model.state_dict(), model_save_path)
    return train_losses, val_losses

train_losses, val_losses = train_model(
    model_package=model_pckg,
    train_loader=train_loader,
    val_loader=None,
    num_epochs=150,
    early_stop_criterion=5,
    record_modifier=0.5,
    record_modifier=0.5,
    model_save_name="transfer_sam.pth",
    save_model=True
)

losses_csv = save_folder + "losses.csv"
csv_file = open(losses_csv, "w")
csv_file.write("train\n")
for loss in train_losses:
    observation = str(loss) + "\n"
    csv_file.write(observation) 

csv_file.write("\n")
csv_file.close()