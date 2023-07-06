import os
import glob
import re 

input_folder = 'data/ags_c11_704x352_24'
files= glob.glob(os.path.join(input_folder,'*.bmp'))

gt_files = []
all_frames = []

for file in files:
    if 'GT_' in os.path.basename(file):
        gt_files.append(file)
    else:
        all_frames.append(file)

all_files = []
for gt_file in gt_files:
    each_data_row = [0,0,0]
    frames = re.findall('GT_\d+_\d+',gt_file)[0]
    frames = frames.split('_')
    # print('sssssssssss',frames)
    
    for frame in all_frames:
        # print(frame)
        frame_num = re.findall('_frame\d+',frame)[0]
        frame_num = frame_num.replace('_frame','')
        # print(frame_num)
        
        if frame_num == frames[1]:
            each_data_row[0] = frame
            to_remove = frame
        elif frame_num == frames[2]:
            each_data_row[1] = frame
    
    each_data_row[2] = gt_file
    # break
    all_files.append(" ".join(each_data_row))
    
all_files = "\n".join(all_files)
    
with open('train.txt', 'w') as f:
   f.write(all_files)
