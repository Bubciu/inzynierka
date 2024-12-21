"""
This script loads videos and runs video_to_file function which processes the videos
and saves them as csv files in directory specified in path_out variable.
"""

from funcs.os_operations import files_in_directory
from funcs.helper_functions import video_to_file

if __name__ == "__main__":

    path_in = "correctness/Vids/sklonBok"
    path_out = "correctness/CSVs/sklonBok"

    names = files_in_directory(path_in, '.mp4')

    for name in names:
        # if name[:8] == 'brzuszek' or name[:3] == 'nic' or name[:5] == 'pajac' or name[:8] == 'przysiad':
            # continue
        print(f"{name}")
        # print("default: ", end="\t")
        video_to_file(rf"{path_in}/{name}", rf"{path_out}/{name[:-4]}.csv", flip=False, show_video=False)
        # # print("flipped: ", end="\t")
        video_to_file(rf"{path_in}/{name}", rf"{path_out}/{name[:-4]}_f.csv", flip=True, show_video=False)
        # # print("")
