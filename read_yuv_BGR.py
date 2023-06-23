import os
import numpy as np
import cv2


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

def  extract_frames_from_yuv_video(yuv_file_path, output_folder_path):
    """
    This function extracts frames from a.yuv video file with RGB information and saves the frames in a specific folder.

    Parameters:
    yuv_file_path (str): The path to the .yuv video file.
    output_folder_path (str): The path to the folder where the extracted frames will be saved.
    """

    try:
        # Check if the input file exists
        if not os.path.isfile(yuv_file_path):
            raise FileNotFoundError("The input file does not exist")
        
        # Check if the output folder exists, if not, create it
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)

        # Open the .yuv file and extract the frames
        with open(yuv_file_path, "rb") as yuv_file:
            prefix, w, h, fps = extract_data_from_file_name(yuv_file_path)
            frame_size = int(w * h * 1.5)
            frame_count = 0
            print("prefix=" + prefix)
            print("width=" + str(w))
            print("height=" + str(h))
            print("framesize= " + str(frame_size))
            print("framerate=" + str(fps))
            frame_count = 0
            while True:
                # Read the next frame from the file
                frame_data = yuv_file.read(frame_size)
                if not frame_data:
                    break

                # Convert the frame data to a numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((int(h * 1.5), w))

                # Convert the YUV frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

                # Save the frame as a PNG/BMP file
                frame_path = os.path.join(output_folder_path, f"frame_{frame_count:04d}.bmp")
                cv2.imwrite(frame_path, frame)

                frame_count +=1

    except Exception as e:
        # Log the error
        print(f"Error: {e}")


def main():
    input_file_path = "./Amazon_1280x720_30/Amazon_1280x720_30.yuv"
    output_folder = "./Amazon_1280x720_30/frames"
    extract_frames_from_yuv_video(input_file_path, output_folder)

if __name__ == "__main__":
    main()