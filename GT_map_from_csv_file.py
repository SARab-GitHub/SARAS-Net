import csv
import numpy as np
from PIL import Image
import os



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


def read_csv_file(file_path, video_name):
    """
    This function reads a .csv file and returns the frame numbers and binary values for a specific video name.

    Parameters:
    file_path (str): The path to the .csv file
    video_name (str): The name of the video to select

    Returns:
    list: A list of tuples containing the frames numbers and binary GT for the selected video name
    """

    try:
        # Open the .csv file
        with open(file_path, 'r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)

            # Initialize an empty list to store the frame numbers and GT
            video_frames = []

            # Iterate over each row in the .csv file
            for row in csv_reader:
                # Check if the video name matches the selected video name
                if row[0] == video_name:
                    # Extract the frame number and GT
                    frame_number = int(row[1])
                    GT = int(row[2])

                    # Append the fram number and binart value as a tuple to the video_frames list
                    video_frames.append((frame_number, GT))

            # Return the list of video frames
            return np.asarray(video_frames)
        
    except FileNotFoundError:
        # Log the error
        print(f"Error: File '{file_path}' not found")
        return []


def generate_image_from_GT(array, width, height, file_name, output_path):
    """
    This function reads pairs of sequential rows from a 2D array. For each pair, if the second element of the second row
    has a value of 1, it generates a wxh all white .bmp image. Otherwise, it generates a w h all black .bmp image.
    It saves these images

    Parameters:
    array (list): The 2D array containing frames number and GT
    width (int): width of the generated image
    height (int): height of the generated image
    file_name (str): file name


    Returns:
    None
    """
        
    # Iterate through pairs of sequential rows
    for i in range(len(array) -1):
        # current_row = array[i]
        next_row = array[i + 1]

        # Check if the GT of the second row is 1
        if len(next_row) >= 2 and next_row[1] == 1:
            # Genetare a width by heigth all white image
            image_array = np.ones((height, width), dtype=np.uint8) * 255
        else:
            # Genetare a width by heigth all black image
            image_array = np.zeros((height, width), dtype=np.uint8)


        # print(image_array)
        # Create PIL Image object from the array
        image = Image.fromarray(image_array)

        # Save the image as a .bmp file
        image_path = output_path
        image.save(f"{image_path}/{file_name}_GT_{i}_{i+1}.bmp")


def main():

    csv_file_path = '/home/ec2-user/Dataset/SCD140_SCD_GT.csv'
    vid_dir = "/home/ec2-user/Dataset/raw_videos/"
    destination_dir = "/home/ec2-user/Dataset/raw_frames/"

    for vid in os.listdir(vid_dir):
        vid_name = vid.split('.')[0]
        GT_output_path = os.path.join(destination_dir, vid_name)
        prefix, h, w, fps = extract_data_from_file_name(vid)
        GT_array = read_csv_file(csv_file_path, vid_name)
        generate_image_from_GT(GT_array, h, w, prefix, GT_output_path)


if __name__ == "__main__":
    main()
