# %%
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

mask_folder = "../consep_masks"
hist_folder = "../consep"
# sort the files by name
hist_files = sorted(os.listdir(hist_folder))
for hist_file in hist_files:
    file_name = hist_file.split(".")[0]
    mask_file = os.path.join(mask_folder, file_name + "_mask.png")
    if os.path.exists(mask_file):
        print(file_name, mask_file)
        break
# %%
# plot the first 10 images
for hist_file in hist_files[:10]:
    file_name = hist_file.split(".")[0]
    mask_file = os.path.join(mask_folder, file_name + "_mask.png")
    if os.path.exists(mask_file):
        hist_image = plt.imread(os.path.join(hist_folder, hist_file))
        mask_image = plt.imread(mask_file)
        bg_img = (hist_image * 255).clip(0, 255).astype(np.uint8)
        overlap_img = bg_img.copy()

        # 2. Extract Contours
        # Ensure mask is 2D uint8 (H, W)
        binary_mask = (mask_image > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Draw Contours (Color is BGR, so (0, 255, 0) is Green)
        cv2.drawContours(overlap_img, contours, -1, (0, 255, 0), 1)

        # 4. Plot
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(bg_img)
        ax[0].set_title("Original Hist")
        
        ax[1].imshow(binary_mask, cmap='gray')
        ax[1].set_title("Binary Mask")
        
        ax[2].imshow(overlap_img)
        ax[2].set_title("Overlap (Contours)")
        plt.show()
        #break
# %%
