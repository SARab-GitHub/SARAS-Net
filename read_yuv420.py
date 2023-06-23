import numpy as np
import cv2
import os


def extract_data_from_file_name(fname):
    """
    Extracts informations regarding the video file from the name of .yuv file
    
    """
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

def read_frame_yuv420(fp, frm_ind, w, h):
    fp.seek(frm_ind * (int(w * h * 3) // 2), 0)
    img_y = np.fromfile(fp, np.uint8, w * h)
    img_u = np.fromfile(fp, np.uint8, int(w // 2) * int(h // 2))
    img_v = np.fromfile(fp, np.uint8, int(w // 2) * int(h // 2))
    img_y = img_y.reshape((h, w))
    img_u = img_u.reshape((int(h//2), int(w//2)))
    img_v = img_v.reshape((int(h//2), int(w//2)))
    return img_y, img_u, img_v

def main():
    input = "/home/gta/Documents/MediaDelivery/6days_7nights_a1_640x272_24.yuv"
    prefix, w, h, fps = extract_data_from_file_name(input)
    print("prefix=" + prefix)
    print("width=" + str(w))
    print("height=" + str(h))
    print("framerate=" + str(fps))

    nframes = os.path.getsize(input) // (int(w * h * 3) // 2)
    print("nframes=" + str(nframes))

    fp = open(input, "rb")
    for i in range(nframes):

        #read frame
        img_y, img_u, img_v = read_frame_yuv420(fp, i, w, h)

        # dump Luma frame into bmp
        cv2.imwrite("frame" + str(i) + ".png", img_y)
        print("frame " + str(i) + " luma channel written as PNG image")
    return 0

if __name__ == "__main__":
    main()