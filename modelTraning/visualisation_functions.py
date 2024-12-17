import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualisation_plot(landmarks: np.ndarray,
                       output_image_name: str = "output_plot.png",
                       save_path: str = ".\\",
                       show: bool = False) -> None:
    """
    Function creates image (.png) from provided array containing landmarks' x and y coordinates.
    :param landmarks: 3-dimensional numpy array with landmarks' coordinates in frames
    :param output_image_name: Name of the output image. If not provided, the plot will be saved as "output_plot.png"
    :param save_path: Full path to a directory to where image should be saved.
        If not provided, the image will be saved in current directory.
    :param show: Specifies if the image should be shown.
    :return: None
    """

    fig, axs = plt.subplots(1, 2, figsize=(3, 3))
    plt.tight_layout(pad=0)

    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)

    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)

    axs[0].imshow(landmarks[:, :, 0])
    axs[1].imshow(landmarks[:, :, 1])

    canvas = fig.canvas

    canvas.draw()

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)

    plt.imsave(rf"{save_path}\{output_image_name}", image)
    plt.close()


def ndarray_to_image(landmarks: np.ndarray, show: bool = False) -> np.ndarray:
    """
    Function creates image (.png) from provided array containing landmarks' x and y coordinates.
    :param landmarks: 3-dimensional numpy array with landmarks' coordinates in frames
    :param show: Specifies if the image should be shown.
    :return: Image with pixel colour in rage <0, 1>.
    """

    fig, axs = plt.subplots(1, 2, figsize=(3, 3))
    plt.tight_layout(pad=0)

    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)

    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)

    axs[0].imshow(landmarks[:, :, 0])
    axs[1].imshow(landmarks[:, :, 1])

    canvas = fig.canvas

    canvas.draw()

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)

    if show:
        # cv2.imshow('win', image)
        # cv2.waitKey(0)
        plt.show()
    else:
        fig.clear()
        plt.close()

    return image / 255.0


def visualisation_video(landmarks: np.ndarray,
                        output_video_name: str = "output_video.avi",
                        save_path: str = ".\\",
                        n_fps: int = 15,
                        height: int = 720,
                        width: int = 1280) -> None:
    """
    Function creates video (.avi) from provided array containing landmarks' x and y coordinates with
    specified width, height and fps. The landmarks are drawn as circles.
    :param landmarks: 3-dimensional numpy array with landmarks' coordinates in frames
    :param output_video_name: Name of the output video. If not provided, the video will be saved as "output_video.avi"
    :param save_path: Full path to a directory to where video should be saved.
        If not provided, the video will be saved in current directory.
    :param n_fps: number of fps
    :param height: frame height in pixels
    :param width:  frame width in pixels
    :return: None
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(fr"{save_path}\{output_video_name}", fourcc, n_fps, (height, width))

    for frame_landmarks in landmarks:
        frame = np.ones((width, height, 3), dtype=np.int8) * 50

        for landmark in frame_landmarks:
            # x, y = landmark
            cv2.circle(frame, (int(round(landmark[0])), int(round(landmark[1]))), 6, (0, 255, 0), -1)

        video_writer.write(frame)

    video_writer.release()
