import cv2
import numpy as np
from utils import *


def draw_unlabeled_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = get_corners(box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


class IDInputWindow:
    def __init__(self):
        self.current_id = ""
        self.input_complete = False
        self.cancelled = False

    def get_player_id(self, x, y):
        """Get player ID through GUI input"""
        self.current_id = ""
        self.input_complete = False
        self.cancelled = False

        # create input window
        input_img = np.zeros((150, 400, 3), dtype=np.uint8)
        cv2.putText(
            input_img,
            "Enter Player ID:",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            input_img,
            "Type ID and press ENTER",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            input_img,
            "Press ESC to cancel",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.namedWindow("Player ID Input", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Player ID Input", input_img)

        while not self.input_complete and not self.cancelled:
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # ENTER
                if self.current_id.strip():
                    self.input_complete = True
            elif key == 27:  # ESC
                self.cancelled = True
            elif key == 8:  # BACKSPACE
                self.current_id = self.current_id[:-1]
            elif 32 <= key <= 126:  # printable characters
                self.current_id += chr(key)

            input_img = np.zeros((150, 400, 3), dtype=np.uint8)
            cv2.putText(
                input_img,
                "Enter Player ID:",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                input_img,
                f"ID: {self.current_id}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                input_img,
                "Type ID and press ENTER",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                input_img,
                "Press ESC to cancel",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            cv2.imshow("Player ID Input", input_img)

        cv2.destroyWindow("Player ID Input")

        if self.cancelled:
            return None
        return self.current_id.strip()


def on_mouse_press(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        box_to_remove = -1

        for i, box in enumerate(param["unlabeled_boxes"]):
            x1, y1, x2, y2 = get_corners(box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"Clicked on box at ({x},{y})")

                id_input = IDInputWindow()
                player_id = id_input.get_player_id(x, y)

                if player_id is not None:  # user didn't cancel
                    param["manual_ids"].append(player_id)
                    param["selected_boxes"].append(box)
                    box_to_remove = i
                    cv2.rectangle(param["frame"], (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        param["frame"],
                        f"ID {player_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Label Frame", param["frame"])
                break

        if box_to_remove > -1:
            param["unlabeled_boxes"].pop(box_to_remove)


def run_user_labeling(annotated_frame, unlabeled_boxes, param):
    draw_unlabeled_boxes(annotated_frame, unlabeled_boxes)

    cv2.putText(
        annotated_frame,
        "Click on players to assign IDs",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        "Press SPACE when done, ESC to cancel",
        (10, annotated_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Label Frame", annotated_frame)
    cv2.setMouseCallback("Label Frame", on_mouse_press, param=param)

    while True:
        # TODO: add break when num_players labeled
        key = cv2.waitKey(30) & 0xFF
        if key == 32:  # SPACE - done labeling
            break
        elif key == 27:  # ESC - cancel
            param["manual_ids"].clear()
            param["selected_boxes"].clear()
            break

    cv2.destroyAllWindows()


class VideoPlayerLabeler:
    def __init__(self, detector, vid_src):
        self.detector = detector
        self.vid_src = vid_src
        self.cap = cv2.VideoCapture(vid_src)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        self.playing = False

        self.selected_frame_num = None
        self.selected_boxes = []
        self.sv_ids = []
        self.unlabeled_boxes = []

        cv2.namedWindow("Video Player - Select Frame for Labeling", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(
            "Frame",
            "Video Player - Select Frame for Labeling",
            0,
            self.frame_count - 1,
            self.on_trackbar_change,
        )

    def on_trackbar_change(self, val):
        self.current_frame = val
        self.display_frame()

    def display_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"Frame: {self.current_frame}/{self.frame_count-1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                "Controls: SPACE=Play/Pause, ENTER=Select Frame, ESC=Exit",
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if self.playing:
                cv2.putText(
                    display_frame,
                    "PLAYING",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display_frame,
                    "PAUSED",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Video Player - Select Frame for Labeling", display_frame)

    def play_video(self):
        """Main video player loop"""
        self.display_frame()

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC - exit
                break
            elif key == 32:  # SPACE - play/pause
                self.playing = not self.playing
                self.display_frame()
            elif key == 13:  # ENTER - select current frame for labeling
                self.selected_frame_num = self.current_frame
                cv2.destroyWindow("Video Player - Select Frame for Labeling")
                return self.start_labeling()

            # auto-advance frame if playing
            if self.playing:
                self.current_frame = min(self.current_frame + 1, self.frame_count - 1)
                cv2.setTrackbarPos(
                    "Frame",
                    "Video Player - Select Frame for Labeling",
                    self.current_frame,
                )
                self.display_frame()

                # pause at end
                if self.current_frame >= self.frame_count - 1:
                    self.playing = False

        cv2.destroyAllWindows()
        self.cap.release()
        return [], [], []

    def start_labeling(self):
        """Start the labeling process for the selected frame"""
        frame = get_frame_from_vid(self.vid_src, self.selected_frame_num)
        if frame is None:
            print(f"Failed to get frame {self.selected_frame_num}")
            return [], [], []

        result = self.detector.detect_frame(frame)
        self.unlabeled_boxes = get_person_boxes(result)
        self.selected_boxes = []
        self.sv_ids = []
        annotated_frame = frame.copy()

        param_map = {
            "unlabeled_boxes": self.unlabeled_boxes,
            "selected_boxes": self.selected_boxes,
            "manual_ids": self.sv_ids,
            "frame": annotated_frame,
        }

        run_user_labeling(annotated_frame, self.unlabeled_boxes, param_map)

        crops = [crop_frame(box, frame) for box in self.selected_boxes]
        self.cap.release()

        return self.sv_ids, crops, self.selected_boxes


def supervised_label(detector, vid_src):
    """Return a list of tuples (id, crop) pairing the user-labeled crops
    of the user-chosen reference frame with the respective user-entered IDs."""

    player = VideoPlayerLabeler(detector, vid_src)
    return player.play_video()


def label_seed_boxes(detector, vid_src):
    """Run the manual labeling UI and return (labels, seed_boxes, ref_frame_idx) for
    the SAM 2 backbone: scrub to a frame where the players are visible, click each
    one, type an id. `labels[k]` is the user id for `seed_boxes[k]` (xyxy)."""
    player = VideoPlayerLabeler(detector, vid_src)
    sv_ids, _crops, boxes = player.play_video()
    ref = player.selected_frame_num if player.selected_frame_num is not None else 0
    seeds = [list(get_corners(box)) for box in boxes]
    return sv_ids, seeds, ref


def main():
    return


if __name__ == "__main__":
    main()
