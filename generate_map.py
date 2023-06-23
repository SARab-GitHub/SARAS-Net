import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.datasets as dates
from torch.autograd import Variable
from torch.nn import functional as F
import shutil
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from torchvision.utils import save_image
import model as models
import glob    
import torchvision.transforms.functional as TF


# Select image 
# yuv_file_name = '6days_7nights_a1_640x272_24.yuv'             
# img1_file = './scene_change_det/frame_0069.bmp'
# img2_file = './scene_change_det/frame_0070.bmp'


# Load model and weight
# pretrain_deeplab_path = "./model_weight_LEVIR.pth"

# Output directory
output_dir = "./dataset/6days_7nights_a1_640x272_24/change_map"


def extract_data_from_file_name(fname):
    parts = fname.split('_')
    parts_len = len(parts)
    fps = parts[parts_len - 1].split('.')[0]
    w = parts[parts_len - 2].split('x')[0]
    h = parts[parts_len - 2].split('x')[1]
    parts = fname.split('\\')
    parts_len = len(parts)
    prefix = parts[parts_len - 1]
    parts = prefix.split("_" + w + "x" + h + "_")
    prefix = parts[0]
    return prefix, int(w), int(h), int(fps)


def get_directory_info(directory_path):
    """
    This function takes a directory path as an argument and returns the number of elements in the directory,
    a list of all file names in the directory, and a list of all file paths in the directory.

    Parameters:
    directory_path (str): The path of the directory to be analyzed
      """
    try:
        # Check if the directory exists
        if not os.path.isdir(directory_path):
            raise ValueError("The specified path is not a directory")
        
        # Get the number of elements in the directory
        num_elements = len(os.listdir(directory_path))

        # Get a list of all file paths in the directory
        file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        file_names.sort()

        # Get a list of all file paths in the directory
        file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        file_paths.sort()

        return num_elements, file_names, file_paths
    except ValueError as e:
        # Log the error
        print(f"Error:  {e}")
        return 0, [], []



def cal_map(input1_path, input2_path, yuv_file_name, output_path, model_path="./model_weight_LEVIR.pth"):
    file1_name_no_ext = os.path.splitext(os.path.basename(input1_path))[0]
    file2_name_no_ext = os.path.splitext(os.path.basename(input2_path))[0]

    prefix, w, h, fps = extract_data_from_file_name(yuv_file_name)
    # print("prefix=" + prefix)
    # print("width=" + str(w))
    # print("height=" + str(h))
    # print("framerate=" + str(fps))

    # Set device and model
    device = torch.device("cpu")
    model = models.Change_detection()
    model = nn.DataParallel(model)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()

    # Pre-process the input images
    img1 = Image.open(input1_path)
    img2 = Image.open(input2_path)

    temp_img1 = img1.resize((w, h))
    temp_img2 = img2.resize((w, h))


    height,width,_ = np.array(temp_img1,dtype= np.uint8).shape
    temp_img1 = np.array(temp_img1,dtype= np.uint8)
    temp_img2 = np.array(temp_img2,dtype= np.uint8)

    temp_img1 = TF.to_tensor(temp_img1)                                          
    temp_img2 = TF.to_tensor(temp_img2)       

    temp_img1 = TF.normalize(temp_img1, mean=[0.44758545, 0.44381796,  0.37912835],std=[0.21713617, 0.20354738, 0.18588887])   
    temp_img2 = TF.normalize(temp_img2, mean=[0.34384388, 0.33675833, 0.28733085],std=[0.1574003, 0.15169171, 0.14402839])  

    label = np.zeros((h,w,3),dtype=np.uint8)
    label = torch.from_numpy(label).long()                                       

    inputs1,input2, targets = temp_img1, temp_img2, label
    inputs1,input2,targets = inputs1.to(device),input2.to(device), targets.to(device)
    # inputs1,inputs2,targets = Variable(inputs1.unsqueeze(0), volatile=True),Variable(input2.unsqueeze(0),volatile=True) ,Variable(targets)
    inputs1,inputs2,targets = Variable(inputs1.unsqueeze(0)),Variable(input2.unsqueeze(0)) ,Variable(targets)


    # model
    output_map = model(inputs1,inputs2)
    output_map = output_map.detach()

    output_map[:,1,:,:] = output_map[:,1,:,:] 
    pred = output_map.argmax(dim=1, keepdim=True)
    pred = pred.cpu().detach().numpy()
    pred_acc = pred 
    pred = pred.squeeze()
    pred = pred*255

    # output
    output_file_name = "map_" + file1_name_no_ext + "_" + file2_name_no_ext + ".bmp"
    # cv2.imwrite(output_path + "pred.jpg", pred)
    cv2.imwrite(os.path.join(output_path, output_file_name), pred)
    print("Change map for: {} - {}".format(file1_name_no_ext, file2_name_no_ext))



if __name__ == "__main__":

    yuv_file_name = '6days_7nights_a1_640x272_24.yuv'  
    output_dir = "./dataset/6days_7nights_a1_640x272_24/change_map"

    video_dir_path = "./dataset/6days_7nights_a1_640x272_24/frames"
    n_elem, name_elem, path_elem = get_directory_info(video_dir_path)

    # print("Number of Elements: ", n_elem)
    # print("List of elements: ", name_elem)
    # print("Element path: ", path_elem)

    for i in range(0, n_elem - 1):
        cal_map(path_elem[i], path_elem[i+1], yuv_file_name, output_dir)
