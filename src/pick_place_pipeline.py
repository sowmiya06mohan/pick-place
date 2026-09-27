import cv2
import sys
import os
from ultralytics import YOLO

# =========================================================
# 1. PATHS
# =========================================================

POSE_MODEL = "yolo11n-pose.pt"
OBJECT_MODEL = "yolo11n.pt"

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "input/pick_place.mp4"

OUTPUT_PATH = "output/final_pick_place.mp4"

os.makedirs("output", exist_ok=True)


# =========================================================
# 2. SETTINGS
# =========================================================

POSE_CONF = 0.30
BACKPACK_CONF = 0.30

ROI_MARGIN = 40

# Number of frames required to confirm
# the initial backpack position.
STABLE_FRAMES = 8

# Number of frames required to confirm
# a state transition.
STATE_FRAMES = 3

# Wrist confidence threshold.
WRIST_CONF = 0.40

# Minimum backpack movement from its
# original position in pixels.
MOVE_DISTANCE = 50


# =========================================================
# 3. LOAD MODELS
# =========================================================

pose_model = YOLO(POSE_MODEL)

object_model = YOLO(OBJECT_MODEL)


# =========================================================
# 4. OPEN INPUT VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("Error: Could not open video.")

    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# =========================================================
# 5. CREATE OUTPUT VIDEO
# =========================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# =========================================================
# 6. ROI VARIABLES
# =========================================================

roi = None

roi_initialized = False

candidate_box = None

stable_count = 0

# Original backpack center.
original_backpack_center = None


# =========================================================
# 7. STATE VARIABLES
# =========================================================

state = "PLACED"

state_counter = 0

# This becomes True after the first complete
# pick-and-place event.
event_completed = False


# =========================================================
# 8. FRAME DETECTION VARIABLES
# =========================================================

person_present = False

backpack_visible = False

backpack_inside_roi = False

wrist_inside = False


# =========================================================
# 9. EVENT COUNTERS
# =========================================================

backpack_moved_counter = 0

person_absent_counter = 0

person_return_counter = 0

backpack_inside_counter = 0


# =========================================================
# 10. STATE CHANGE FUNCTION
# =========================================================

def change_state(new_state):

    global state

    global state_counter

    if state != new_state:

        print(
            f"STATE CHANGE: {state} -> {new_state}"
        )

        state = new_state

        state_counter = 0


# =========================================================
# 11. MAIN VIDEO LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:

        break


    # =====================================================
    # RESET CURRENT FRAME VARIABLES
    # =====================================================

    person_present = False

    backpack_visible = False

    backpack_inside_roi = False

    wrist_inside = False


    # =====================================================
    # PERSON POSE + TRACKING
    # =====================================================

    pose_results = pose_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=POSE_CONF,
        verbose=False
    )

    pose_result = pose_results[0]

    annotated_frame = pose_result.plot()


    # =====================================================
    # PERSON DETECTION
    # =====================================================

    if pose_result.boxes is not None:

        if len(pose_result.boxes) > 0:

            person_present = True


    # =====================================================
    # WRIST / POSE KEYPOINT DETECTION
    # =====================================================

    if (
        person_present
        and
        pose_result.keypoints is not None
        and
        len(pose_result.keypoints) > 0
    ):

        keypoints = (
            pose_result.keypoints.xy.cpu().numpy()
        )


        keypoint_conf = None


        if pose_result.keypoints.conf is not None:

            keypoint_conf = (
                pose_result.keypoints.conf.cpu().numpy()
            )


        # Use first detected person.
        person_points = keypoints[0]


        # COCO pose keypoints:
        # 9  = left wrist
        # 10 = right wrist

        left_x = int(
            person_points[9][0]
        )

        left_y = int(
            person_points[9][1]
        )

        right_x = int(
            person_points[10][0]
        )

        right_y = int(
            person_points[10][1]
        )


        left_valid = True

        right_valid = True


        if keypoint_conf is not None:

            left_valid = (
                keypoint_conf[0][9]
                >= WRIST_CONF
            )

            right_valid = (
                keypoint_conf[0][10]
                >= WRIST_CONF
            )


        # -------------------------------------------------
        # DRAW LEFT WRIST
        # -------------------------------------------------

        if left_valid:

            cv2.circle(
                annotated_frame,
                (left_x, left_y),
                7,
                (0, 255, 0),
                -1
            )


            cv2.putText(
                annotated_frame,
                "L Wrist",
                (left_x + 5, left_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


        # -------------------------------------------------
        # DRAW RIGHT WRIST
        # -------------------------------------------------

        if right_valid:

            cv2.circle(
                annotated_frame,
                (right_x, right_y),
                7,
                (0, 255, 0),
                -1
            )


            cv2.putText(
                annotated_frame,
                "R Wrist",
                (right_x + 5, right_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


        # =================================================
        # CHECK WRIST INSIDE ROI
        # =================================================

        if roi is not None:

            rx1, ry1, rx2, ry2 = roi


            left_inside = (
                left_valid
                and
                rx1 <= left_x <= rx2
                and
                ry1 <= left_y <= ry2
            )


            right_inside = (
                right_valid
                and
                rx1 <= right_x <= rx2
                and
                ry1 <= right_y <= ry2
            )


            wrist_inside = (
                left_inside
                or
                right_inside
            )


    # =====================================================
    # BACKPACK DETECTION
    # =====================================================

    object_results = object_model(
        frame,
        conf=BACKPACK_CONF,
        verbose=False
    )

    object_result = object_results[0]


    backpack_box = None

    backpack_conf = 0.0


    if object_result.boxes is not None:

        best_confidence = 0.0

        best_box = None


        for box in object_result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = (
                object_model.names[class_id]
            )


            # Only accept the backpack class.
            if (
                class_name == "backpack"
                and
                confidence >= BACKPACK_CONF
            ):

                if confidence > best_confidence:

                    best_confidence = confidence

                    best_box = box


        if best_box is not None:

            x1, y1, x2, y2 = map(
                int,
                best_box.xyxy[0]
            )


            backpack_box = (
                x1,
                y1,
                x2,
                y2
            )


            backpack_conf = (
                best_confidence
            )


    backpack_visible = (
        backpack_box is not None
    )


    # =====================================================
    # INITIALIZE STABLE BACKPACK ROI
    # =====================================================

    if (
        backpack_box is not None
        and
        not roi_initialized
    ):

        x1, y1, x2, y2 = backpack_box


        if candidate_box is None:

            candidate_box = backpack_box

            stable_count = 1


        else:

            old_cx = (
                candidate_box[0]
                +
                candidate_box[2]
            ) / 2


            old_cy = (
                candidate_box[1]
                +
                candidate_box[3]
            ) / 2


            new_cx = (
                x1 + x2
            ) / 2


            new_cy = (
                y1 + y2
            ) / 2


            distance = (
                (new_cx - old_cx) ** 2
                +
                (new_cy - old_cy) ** 2
            ) ** 0.5


            if distance < 40:

                stable_count += 1

            else:

                candidate_box = backpack_box

                stable_count = 1


        # -------------------------------------------------
        # CREATE ROI
        # -------------------------------------------------

        if stable_count >= STABLE_FRAMES:

            x1, y1, x2, y2 = candidate_box


            roi = (
                max(0, x1 - ROI_MARGIN),
                max(0, y1 - ROI_MARGIN),
                min(width, x2 + ROI_MARGIN),
                min(height, y2 + ROI_MARGIN)
            )


            # Save original backpack center.
            original_backpack_center = (
                (x1 + x2) / 2,
                (y1 + y2) / 2
            )


            roi_initialized = True


            print(
                "Stable backpack detected."
            )


            print(
                "ROI initialized:",
                roi
            )


            print(
                "Original backpack center:",
                original_backpack_center
            )


    # =====================================================
    # BACKPACK POSITION CHECK
    # =====================================================

    backpack_moved_distance = 0.0


    if (
        backpack_box is not None
        and
        roi is not None
    ):

        x1, y1, x2, y2 = backpack_box


        backpack_center_x = (
            x1 + x2
        ) / 2


        backpack_center_y = (
            y1 + y2
        ) / 2


        rx1, ry1, rx2, ry2 = roi


        # -------------------------------------------------
        # IS BACKPACK INSIDE ORIGINAL ROI?
        # -------------------------------------------------

        backpack_inside_roi = (
            rx1 <= backpack_center_x <= rx2
            and
            ry1 <= backpack_center_y <= ry2
        )


        # -------------------------------------------------
        # DISTANCE FROM ORIGINAL LOCATION
        # -------------------------------------------------

        if original_backpack_center is not None:

            original_x = (
                original_backpack_center[0]
            )

            original_y = (
                original_backpack_center[1]
            )


            backpack_moved_distance = (
                (
                    (backpack_center_x - original_x)
                    ** 2
                    +
                    (backpack_center_y - original_y)
                    ** 2
                )
                ** 0.5
            )


    # =====================================================
    # DRAW BACKPACK BOX
    # =====================================================

    if backpack_box is not None:

        x1, y1, x2, y2 = backpack_box


        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


        cv2.putText(
            annotated_frame,
            f"Backpack {backpack_conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )


    # =====================================================
    # DRAW ROI
    # =====================================================

    if roi is not None:

        rx1, ry1, rx2, ry2 = roi


        cv2.rectangle(
            annotated_frame,
            (rx1, ry1),
            (rx2, ry2),
            (0, 255, 255),
            2
        )


        cv2.putText(
            annotated_frame,
            "BACKPACK ROI",
            (rx1, max(20, ry1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


    # =====================================================
    # STATE MACHINE
    #
    # PLACED
    #    ↓
    # PICKING
    #    ↓
    # PICKED
    #    ↓
    # PERSON LEFT
    #    ↓
    # PLACING
    #    ↓
    # PLACED
    #
    # After the first complete event, the state machine
    # stops so a second accidental cycle cannot occur.
    # =====================================================

    if (
        roi_initialized
        and
        not event_completed
    ):


        # =================================================
        # 1. PLACED -> PICKING
        # =================================================

        if state == "PLACED":

            if (
                wrist_inside
                and
                backpack_inside_roi
            ):

                state_counter += 1

            else:

                state_counter = 0


            if state_counter >= STATE_FRAMES:

                change_state(
                    "PICKING"
                )


                backpack_moved_counter = 0


        # =================================================
        # 2. PICKING -> PICKED
        # =================================================

        elif state == "PICKING":

            if (
                backpack_visible
                and
                backpack_moved_distance
                >= MOVE_DISTANCE
            ):

                backpack_moved_counter += 1


            # Do not reset immediately when there is
            # a temporary detection fluctuation.


            if (
                backpack_moved_counter
                >= STATE_FRAMES
            ):

                change_state(
                    "PICKED"
                )


                person_absent_counter = 0


        # =================================================
        # 3. PICKED -> PERSON LEFT
        # =================================================

        elif state == "PICKED":

            if not person_present:

                person_absent_counter += 1

            else:

                person_absent_counter = 0


            if person_absent_counter >= 5:

                change_state(
                    "PERSON LEFT"
                )


                person_return_counter = 0


        # =================================================
        # 4. PERSON LEFT -> PLACING
        # =================================================

        elif state == "PERSON LEFT":

            if person_present:

                person_return_counter += 1

            else:

                person_return_counter = 0


            if person_return_counter >= 5:

                change_state(
                    "PLACING"
                )


                backpack_inside_counter = 0


        # =================================================
        # 5. PLACING -> PLACED
        # =================================================

        elif state == "PLACING":

            if (
                backpack_inside_roi
                and
                not wrist_inside
            ):

                backpack_inside_counter += 1

            else:

                backpack_inside_counter = 0


            if (
                backpack_inside_counter
                >= STATE_FRAMES
            ):

                change_state(
                    "PLACED"
                )


                # -----------------------------------------
                # COMPLETE FIRST EVENT
                # -----------------------------------------

                event_completed = True


                print(
                    "COMPLETE PICK-AND-PLACE EVENT DETECTED."
                )


    # =====================================================
    # DISPLAY PERSON STATUS
    # =====================================================

    if person_present:

        person_text = (
            "PERSON PRESENT"
        )

    else:

        person_text = (
            "PERSON NOT DETECTED"
        )


    cv2.putText(
        annotated_frame,
        person_text,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =====================================================
    # DISPLAY BACKPACK STATUS
    # =====================================================

    if backpack_visible:

        backpack_text = (
            "BACKPACK DETECTED"
        )

    else:

        backpack_text = (
            "BACKPACK NOT DETECTED"
        )


    cv2.putText(
        annotated_frame,
        backpack_text,
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =====================================================
    # DISPLAY WRIST STATUS
    # =====================================================

    if wrist_inside:

        wrist_text = (
            "WRIST INSIDE ROI"
        )

        wrist_color = (
            0,
            255,
            0
        )

    else:

        wrist_text = (
            "WRIST OUTSIDE ROI"
        )

        wrist_color = (
            0,
            0,
            255
        )


    cv2.putText(
        annotated_frame,
        wrist_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        wrist_color,
        2
    )


    # =====================================================
    # DISPLAY CURRENT STATE
    # =====================================================

    cv2.putText(
        annotated_frame,
        f"STATE: {state}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # =====================================================
    # DISPLAY BACKPACK MOVEMENT
    # =====================================================

    if roi_initialized:

        cv2.putText(
            annotated_frame,
            f"Move: {backpack_moved_distance:.0f}px",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # =====================================================
    # DISPLAY EVENT COMPLETION
    # =====================================================

    if event_completed:

        cv2.putText(
            annotated_frame,
            "EVENT COMPLETED",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


    # =====================================================
    # SAVE FRAME
    # =====================================================

    out.write(
        annotated_frame
    )


    # =====================================================
    # SHOW FRAME
    # =====================================================

    cv2.imshow(
        "Pick and Place Tracking",
        annotated_frame
    )


    # =====================================================
    # PRESS Q TO STOP
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================================
# 12. RELEASE RESOURCES
# =========================================================

cap.release()

out.release()

cv2.destroyAllWindows()


# =========================================================
# 13. FINAL OUTPUT
# =========================================================

print()

print(
    "=============================================="
)

print(
    "Pick and Place State Machine Completed"
)

print(
    "=============================================="
)

print(
    f"Final state: {state}"
)

print(
    f"Event completed: {event_completed}"
)

print(
    f"Output saved to: {OUTPUT_PATH}"
)