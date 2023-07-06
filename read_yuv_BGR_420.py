import os
import re
import numpy as np
import cv2



def extract_data_from_file_name(fname):
    """
    This functions reads the name and the data from the video file.

    Parameters:
    fname (str): The name of the video file

    Returns:
    prefix (str): Prefix of the video file
    w (int): width of the video file 
    h (int): height of the video file 
    fps (str): 
    """

    pattern = r"(.+)(_+\d+x\d+_)(.+)"

    parts = re.match(pattern, fname)

    prefix = os.path.basename(parts.group(1))
    second_part = parts.group(2)
    fps = parts.group(3)

    numbers = second_part.split('_')[1]
    w = int(numbers.split('x')[0])
    h = int(numbers.split('x')[1])

    return prefix, w, h, fps




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
            print("framerate=" + str(fps))
            file_name = prefix + '_' + str(w) + 'x' + str(h)
            print()
            while True:
                # Read the next frame from the file
                frame_data = yuv_file.read(frame_size)
                if not frame_data:
                    break

                # Convert the frame data to a numpy array
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape(int(h * 1.5), w)

                # Convert the YUV frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

                # Save the frame as a PNG/BMP file
                frame_name = file_name + '_' + f"frame_{frame_count:04d}.bmp"
                frame_path = os.path.join(output_folder_path, frame_name)
                cv2.imwrite(frame_path, frame)
                print("Written " + frame_name + " in " + output_folder_path)

                frame_count +=1

    except Exception as e:
        # Log the error
        print(f"Error: {e}")


def main():

    source_dir = "/home/ec2-user/Dataset/raw_videos/"
    destination_dir = "/home/ec2-user/Dataset/raw_frames/"

    for element in os.listdir(source_dir):
        element_name = os.path.splitext(os.path.basename(element))[0]
        input_file_path = source_dir + element_name + ".yuv"
        output_folder = destination_dir + element_name
        extract_frames_from_yuv_video(input_file_path, output_folder)

if __name__ == "__main__":
    main()