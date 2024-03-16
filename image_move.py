import os
import shutil

src_fg = './dim_data/fg/'
dest_fg = './dim_data/new_fotor_fg/'

src_matte = './dim_data/matte_fotor/'
src_matte1 = './dim_data/fotor_matte_reduced/'
dest_matte = './dim_data/new_fotor_matte/'

if not os.path.exists(dest_fg):
    os.makedirs(dest_fg)

if not os.path.exists(dest_matte):
    os.makedirs(dest_matte)

for file in os.listdir(src_matte):
    if not (file.endswith(".jpg") or file.endswith(".png")):
        continue
    print(file)
    fg_src_name = os.path.join(src_fg, file)
    matte_src_name = os.path.join(src_matte, file)
    #matte_src_reduced_name = os.path.join(src_matte1, file)

    fg_dest_name = os.path.join(dest_fg, file)
    matte_dest_name = os.path.join(dest_matte, file)
    if not os.path.exists(fg_src_name):
        continue
    
    shutil.copy2(fg_src_name, fg_dest_name)
    shutil.copy2(matte_src_name, matte_dest_name)
    