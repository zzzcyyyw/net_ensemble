import numpy as np
from PIL import Image
import os

path = './dim_data/PhotoMatte85/'
save_path = './dim_data/matte/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
#image = np.array(Image.open(path))   # read in RGB mode
#image = np.asanyarray(image)

for file in os.listdir(path):
    if not (file.endswith(".jpg") or file.endswith(".png")):
        continue
    print(file)
    img_path = os.path.join(path, file)
    image = Image.open(img_path)
    r, g, b, alpha = image.split()

    #image_rgb = Image.merge("RGB", (r, g, b))
    image_alpha = alpha
    #img_rgba2rgb = image.convert("RGB")

    #image_rgb.save("image_rgb.jpg")
    dest_file = os.path.join(save_path, file)
    image_alpha.save(dest_file)
    #img_rgba2rgb.save("img_rgba2rgb.jpg")

    #alpha_arr = np.array(image_alpha)