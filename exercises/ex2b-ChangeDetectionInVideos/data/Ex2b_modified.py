import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)


def capture_from_camera_and_show_images(T = 0.1, A = 0.05, alpha = 0.95):
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = img_as_float(frame_gray)

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)

        # Binary image by applying a threshold to the difference image
        binary_dif_img = dif_img > T

        # Total number of foreground pixels in the binary difference image
        n_foreground_pixels = np.sum(binary_dif_img)

        # Percentage of foreground pixels compared to the total number of pixels in the image
        n_total_pixels = binary_dif_img.size
        F = n_foreground_pixels / n_total_pixels * 100

        # Info about the image
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, f"Num of changed pixels: {n_foreground_pixels}", (100, 200), font, 1, (255, 0, 0), 1)
        cv2.putText(new_frame, f"Avg diff img: {np.mean(dif_img)}", (100, 300), font, 1, (255, 0, 0), 1)
        cv2.putText(new_frame, f"Min value diff img: {np.min(dif_img)}", (100, 400), font, 1, (0, 255, 0), 1)
        cv2.putText(new_frame, f"Max value diff img: {np.max(dif_img)}", (100, 500), font, 1, (0, 255, 0), 1)

        # Raise an alarm if the percentage of foreground pixels F is above a alarm threshold A
        if F > A:
            str_out = "Change detected!"
            cv2.putText(new_frame, str_out, (100, 600), font, 1, (0, 0, 255), 1)

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_frame_gray, 0, 400)
        show_in_moved_window('Difference image', dif_img, 700, 10)
        show_in_moved_window('Binary difference image', img_as_ubyte(binary_dif_img), 700, 400)

        # Old frame is updated
        frame_gray = alpha * frame_gray + (1 - alpha) * new_frame_gray

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    T = 0.2
    A = 0.5
    alpha = 0.95
    capture_from_camera_and_show_images(T, A, alpha)
