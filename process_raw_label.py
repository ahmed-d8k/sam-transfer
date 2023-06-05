import os
import cv2
import numpy as np

# save_folder = "sam_train_demo/train/labels/"
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)

def process_label(img_path, label_color):
    cv_color = (label_color[2], label_color[1], label_color[0])
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img[:,:,:3]
    img = np.where((img[:,:,0] == cv_color[0]) &
                   (img[:,:,1] == cv_color[1]) &
                   (img[:,:,2] == cv_color[2]),
                   (255),
                   (0))
    return img

def save_img(img, save_name, save_location):
    save_path = save_location + "/" + save_name
    cv2.imwrite(save_path, img)

def process_folder(unprocessed_label_path, processed_label_path, label_color):
    img_paths = []
    for root, dirs, files in os.walk(unprocessed_label_path):
        for f in files:
            img_path = os.path.join(root, f)
            label = process_label(img_path, label_color)
            save_img(label, f, processed_label_path)

process_folder("sam_train_demo/train/unprocessed_labels", "sam_train_demo/train/labels", (237, 28, 36))