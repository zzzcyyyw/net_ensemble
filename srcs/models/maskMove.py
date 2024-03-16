import os
import shutil

fg_dir = './data/new_data/fg_fotur/'
mask_dir = './data/new_data/matte_fotur/'
alpha_dir = './data/new_data/fg/'


count = 0
for root, dirs, files in sorted(os.walk(mask_dir)):
    for file in files:
        this_file_path = os.path.join(fg_dir, file)
        if os.path.exists(this_file_path):
            count += 1
            #dest_file_path = os.path.join(alpha_dir, file)
            shutil.copy2(this_file_path, alpha_dir)