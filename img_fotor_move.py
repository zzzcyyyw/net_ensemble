import os
import shutil

src_fg = './data/new_fotor_fg/'
dest_fg = './data/Composite/'

src_matte = './data/new_fotor_matte/'
dest_matte = './data/matte/'

if not os.path.exists(dest_fg):
    os.makedirs(dest_fg)

if not os.path.exists(dest_matte):
    os.makedirs(dest_matte)

for file in os.listdir(src_fg):
    if not (file.endswith(".jpg") or file.endswith(".png")):
        continue
    
    src_matte_file = os.path.join(src_matte, file)
    if not os.path.exists(src_matte_file):
        continue

    print(file)
    #move fg
    dst_fg_file = file + "--Fotor--" + file
    fg_src_name = os.path.join(src_fg, file)
    
    shutil.copy2(fg_src_name, os.path.join(dest_fg, dst_fg_file))

    matte_src_name = os.path.join(src_matte, file)
    #matte_src_reduced_name = os.path.join(src_matte1, file)

    matte_dest_name = os.path.join(dest_matte, file)
    
    #shutil.copy2(fg_src_name, fg_dest_name)
    shutil.copy2(matte_src_name, matte_dest_name)
    