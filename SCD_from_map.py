import os

def calculate_percentage_of_ones(directory_path):
    """
    This function reads all black and white files in a directory and calculates the percentage of 1 values in each file.

    Parameters:
    directory_path (str): The path to the directory containing the files

    Returns:
    dict: A dictionary containing the filename and its corresponding percentage of 1 values
    """

    try:
        # Check if directory exists
        if not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path")
        
        # Get all .jpg files in the directory
        files = [f for f in os.listdir(directory_path) if f.endswith('.bmp')]

        # Calculate percentate of 1 values for each file
        result = {}
        for file in files:
            with open(os.path.join(directory_path, file), 'rb') as f:

                # Reads only the file name without file extension
                file_name = os.path.splitext(os.path.basename(file))[0]

                # Read the file as a string of pixel values
                pixel_values = f.read()

                # Calculate the percentage of 1 values
                num_ones = pixel_values.count(255)
                percentage_ones = (num_ones / len(pixel_values))

                # Add the filename and percentage to the result directory
                result[file_name] = percentage_ones

        return result
    
    except ValueError as e:
        # Log the error
        print(f"Error: {e}")
        return {}
    
def change_detect_bin(change_value_dict):

    detect_change = []
    for k, v in change_value_dict.items():
        if v > 0.0001:
            detect_change.append(1)
        else:
            detect_change.append(0)
    return detect_change

if __name__ == "__main__":

    video_name = "ags_c11_704x352_24"

    # change_map_dir = "./6days_7nights_a1_640x272_24/change_map" 
    change_map_dir = "./" + video_name + "/change_map"

    change_det = dict(sorted(calculate_percentage_of_ones(change_map_dir).items()))
    # print(change_det)
    GT_change = change_detect_bin(change_det)
    change_det_print = "\n".join("{0}, {1}, {2}, {3}, {4}".format(video_name, n, GT_change[n], k, v) for n, (k,v) in enumerate(change_det.items()))
    # print(change_det_print)

    # with open("./6days_7nights_a1_640x272_24/6days_7nights_a1_640x272_24.txt", "w") as text_file:
    with open("./" + video_name + "/" + video_name +".txt", "w") as text_file:
        text_file.write(change_det_print)
