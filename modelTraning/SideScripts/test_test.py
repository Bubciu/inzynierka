from modelTrening.os_operations import files_in_directory
from modelTrening.helper_functions import video_to_file


if __name__ == "__main__":

    path_in = "..\Vids"
    path_out = "..\Bin"

    video_to_file(rf"{path_in}\przysiad20.mp4", rf"{path_out}\{'przysiad20.mp4'[:-4]}.csv", flip=False, show_video=False)

    # names = files_in_directory(path_in, '.mp4')
    #
    # for name in names:
    #     print(f"{name}")
    #     # print("default: ", end="\t")
    #     video_to_file(rf"{path_in}/{name}", rf"{path_out}/{name[:-4]}.csv", flip=False, show_video=False)
    #     # # print("flipped: ", end="\t")
    #     video_to_file(rf"{path_in}/{name}", rf"{path_out}/{name[:-4]}_f.csv", flip=True, show_video=False)
    #     # # print("")