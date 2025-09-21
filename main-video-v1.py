import cv2
import numpy as np
import os


def display_and_split_biggest_rectangle_part(image): # Changed to take image directly
    """
    Loads an image, finds the biggest rectangular part (the OMR form),
    and splits it into the answer section and info section.
    Returns these as NumPy arrays along with their absolute bounding boxes
    and the bounding box of the overall biggest rectangle.
    """
    if image is None:
        print(f"❌ Error: Input image is None.")
        return None, None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_rect = None
    max_area = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 1000:  # Filter by 4 vertices and minimum area
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area > max_area:
                max_area = area
                biggest_rect = (x, y, w, h)

    info_section_abs_bbox = None
    answer_section_img = None
    info_section_img = None

    if biggest_rect:
        x, y, w, h = biggest_rect
        rectangular_part = image[y:y + h, x:x + w]

        # Calculate coordinates for info section within the original image (relative to overall image)
        info_section_with_outer_abs_x = x + int(0.75 * w)
        info_section_with_outer_abs_y = y
        info_section_with_outer_abs_w = w - int(0.75 * w)
        info_section_with_outer_abs_h = h

        answer_section_with_outer = rectangular_part[:, 0:int(0.75 * w)]
        info_section_with_outer = rectangular_part[:, int(0.75 * w):]

        def crop_white_borders_with_offset(section_image, current_abs_x, current_abs_y):
            """
            Finds the main content within a section (removing white borders)
            and returns the cropped image along with its absolute bounding box.
            """
            if section_image is None or section_image.size == 0:
                return section_image, (current_abs_x, current_abs_y, 0, 0)

            gray_section = cv2.cvtColor(section_image, cv2.COLOR_BGR2GRAY)
            _, thresh_section = cv2.threshold(gray_section, 200, 255, cv2.THRESH_BINARY_INV)
            contours_section, _ = cv2.findContours(thresh_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_section:
                largest_contour_section = max(contours_section, key=cv2.contourArea)
                x_sec, y_sec, w_sec, h_sec = cv2.boundingRect(largest_contour_section)

                abs_x = current_abs_x + x_sec
                abs_y = current_abs_y + y_sec
                abs_w = w_sec
                abs_h = h_sec

                return section_image[y_sec:y_sec + h_sec, x_sec:x_sec + w_sec], (abs_x, abs_y, abs_w, abs_h)

            return section_image, (current_abs_x, current_abs_y, section_image.shape[1], section_image.shape[0])

        answer_section_img, _ = crop_white_borders_with_offset(answer_section_with_outer, x, y)
        info_section_img, info_section_abs_bbox = crop_white_borders_with_offset(
            info_section_with_outer, info_section_with_outer_abs_x, info_section_with_outer_abs_y
        )

        # print("✅ Main sections extracted (in memory).") # Too chatty for real-time
        return answer_section_img, info_section_img, info_section_abs_bbox, biggest_rect
    else:
        # print("❌ No significant rectangle found.") # Too chatty for real-time
        return None, None, None, None


def extract_info_sections(info_section_image, info_section_abs_bbox):
    """
    Extracts the 'শিক্ষার্থী আইডি', 'বিষয় কোড', and 'মোট নম্বর' sections
    from the info_section_image (NumPy array) and returns them as NumPy arrays
    along with their absolute bounding boxes in the original image.
    """
    if info_section_image is None or info_section_image.size == 0 or info_section_abs_bbox is None:
        # print("❌ Error: Info section image or its bbox is empty or None.") # Too chatty for real-time
        return None, None, None, None, None, None

    img_height, img_width, _ = info_section_image.shape

    info_abs_x, info_abs_y, _, _ = info_section_abs_bbox

    student_id_section_height_percentage = 0.25
    subject_code_section_height_percentage = 0.25

    student_id_start_y_rel = 0
    student_id_end_y_rel = int(img_height * student_id_section_height_percentage)

    subject_code_start_y_rel = student_id_end_y_rel
    subject_code_end_y_rel = subject_code_start_y_rel + int(img_height * subject_code_section_height_percentage)

    total_marks_start_y_rel = subject_code_end_y_rel
    total_marks_end_y_rel = img_height

    student_id_part = None
    subject_code_part = None
    total_marks_part = None
    student_id_bbox = None
    subject_code_bbox = None
    total_marks_bbox = None

    if student_id_end_y_rel > student_id_start_y_rel:
        student_id_part = info_section_image[student_id_start_y_rel:student_id_end_y_rel, 0:img_width]
        student_id_bbox = (info_abs_x, info_abs_y + student_id_start_y_rel, img_width,
                           student_id_end_y_rel - student_id_start_y_rel)
        # print("✅ Student ID section extracted (in memory).") # Too chatty
    if subject_code_end_y_rel > subject_code_start_y_rel:
        subject_code_part = info_section_image[subject_code_start_y_rel:subject_code_end_y_rel, 0:img_width]
        subject_code_bbox = (info_abs_x, info_abs_y + subject_code_start_y_rel, img_width,
                             subject_code_end_y_rel - subject_code_start_y_rel)
        # print("✅ Subject Code section extracted (in memory).") # Too chatty
    if total_marks_end_y_rel > total_marks_start_y_rel:
        total_marks_part = info_section_image[total_marks_start_y_rel:total_marks_end_y_rel, 0:img_width]
        total_marks_bbox = (info_abs_x, info_abs_y + total_marks_start_y_rel, img_width,
                            total_marks_end_y_rel - total_marks_start_y_rel)
        # print("✅ Total Marks section extracted (in memory).") # Too chatty

    return student_id_part, subject_code_part, total_marks_part, student_id_bbox, subject_code_bbox, total_marks_bbox


def crop_top_percentage(image, percentage):
    """
    Crops the top 'percentage' of an image.
    """
    if image is None or image.size == 0:
        return image

    height, width, _ = image.shape
    crop_height = int(height * percentage)

    if crop_height >= height:
        # print(f"⚠️ Warning: Crop height ({crop_height}) is greater than or equal to image height ({height}). Returning empty array.") # Too chatty
        return np.array([])

    return image[crop_height:height, 0:width]


def split_answer_section_horizontally(answer_section_image, crop_percent_each_column=0.18):
    """
    Splits the answer_section_image (NumPy array) horizontally into two equal halves,
    and then each half into 15 columns. Then, crops the top 'crop_percent_each_column'
    from each of these 30 columns. Returns all these parts as a list of NumPy arrays.
    """
    if answer_section_image is None or answer_section_image.size == 0:
        # print(f"❌ Error: Answer section image is empty or None.") # Too chatty
        return [], []

    img_height, img_width, _ = answer_section_image.shape

    middle_y = img_height // 2

    top_half = answer_section_image[0:middle_y, 0:img_width]
    bottom_half = answer_section_image[middle_y:img_height, 0:img_width]

    # print("✅ Answer section split into two halves (in memory).") # Too chatty

    num_columns = 15
    cropped_top_half_cols = []
    cropped_bottom_half_cols = []

    top_half_cols = np.array_split(top_half, num_columns, axis=1)
    # print(f"✅ Top half split into {len(top_half_cols)} columns.") # Too chatty
    for i, col_img in enumerate(top_half_cols):
        cropped_col = crop_top_percentage(col_img, crop_percent_each_column)
        if cropped_col.size > 0:
            cropped_top_half_cols.append(cropped_col)
        # else:
            # print(f"⚠️ Warning: Top Half Col {i + 1} became empty after {crop_percent_each_column * 100}% top crop.") # Too chatty

    bottom_half_cols = np.array_split(bottom_half, num_columns, axis=1)
    # print(f"✅ Bottom half split into {len(bottom_half_cols)} columns.") # Too chatty
    for i, col_img in enumerate(bottom_half_cols):
        cropped_col = crop_top_percentage(col_img, crop_percent_each_column)
        if cropped_col.size > 0:
            cropped_bottom_half_cols.append(cropped_col)
        # else:
            # print(f"⚠️ Warning: Bottom Half Col {i + 1} became empty after {crop_percent_each_column * 100}% top crop.") # Too chatty

    return cropped_top_half_cols, cropped_bottom_half_cols


def split_and_count_black_pixels_in_parts(column_image, num_parts=4, black_pixel_threshold=50):
    """
    Splits a single column image horizontally into 'num_parts' equal parts
    and counts the number of 'black' pixels in each part.
    'black_pixel_threshold' defines what intensity value is considered black (0-255).
    """
    if column_image is None or column_image.size == 0:
        # print("❌ Error: Empty column image provided for splitting and counting.") # Too chatty
        return []

    gray_column = cv2.cvtColor(column_image, cv2.COLOR_BGR2GRAY)

    height, width = gray_column.shape
    part_height = height // num_parts
    black_pixel_counts = []

    for i in range(num_parts):
        start_y = i * part_height
        end_y = (i + 1) * part_height

        if i == num_parts - 1:
            end_y = height

        part = gray_column[start_y:end_y, 0:width]

        if part.size > 0:
            black_pixels = np.sum(part < black_pixel_threshold)
            black_pixel_counts.append(int(black_pixels))
        else:
            black_pixel_counts.append(0)

    return black_pixel_counts


def get_answer_from_counts(pixel_counts, sensitivity_threshold=20):
    """
    Determines the selected answer (1-4) based on black pixel counts.
    Assumes the highest count indicates the answer.
    'sensitivity_threshold' is the minimum pixel count to consider a bubble filled.
    """
    if not pixel_counts:
        return None

    max_pixels = 0
    selected_answer_index = -1

    # Track how many options are above threshold to detect ambiguous marks
    filled_bubbles_count = 0
    for count in pixel_counts:
        if count >= sensitivity_threshold:
            filled_bubbles_count += 1

    if filled_bubbles_count > 1:  # More than one bubble filled significantly
        return None  # Ambiguous mark

    for i, count in enumerate(pixel_counts):
        if count > max_pixels and count >= sensitivity_threshold:
            max_pixels = count
            selected_answer_index = i

    if selected_answer_index != -1:
        return selected_answer_index + 1  # Convert 0-indexed to 1-indexed answer
    else:
        return None  # Unmarked or no bubble met sensitivity threshold


def get_all_bubble_bboxes(omr_image_shape, biggest_rect, num_questions=30, options_per_question=4,
                          crop_percent_each_column=0.18):
    """
    Calculates the absolute bounding boxes for all answer options on the original image.
    These are regions where we expect to find the bubbles.

    Args:
        omr_image_shape (tuple): Shape (height, width, channels) of the original OMR image.
        biggest_rect (tuple): (x, y, w, h) of the main OMR form rectangle.
        num_questions (int): Total number of questions (e.g., 30).
        options_per_question (int): Number of options per question (e.g., 4).
        crop_percent_each_column (float): Percentage to crop from the top of each column.

    Returns:
        list of lists: all_bubble_bboxes[question_index][option_index] = (x, y, w, h)
                       All coordinates are absolute on the original image. These are
                       bounding boxes for the *area* where a bubble is expected.
    """
    omr_x, omr_y, omr_w, omr_h = biggest_rect

    ans_sec_abs_x = omr_x
    ans_sec_abs_y = omr_y
    ans_sec_abs_w = int(0.75 * omr_w)
    ans_sec_abs_h = omr_h

    all_bubble_bboxes = []
    num_columns_per_half = num_questions // 2

    half_height = ans_sec_abs_h // 2
    column_width = ans_sec_abs_w // num_columns_per_half

    for col_idx in range(num_columns_per_half):
        col_abs_x = ans_sec_abs_x + col_idx * column_width

        crop_pixels_from_top = int(half_height * crop_percent_each_column)
        cropped_col_abs_y = ans_sec_abs_y + crop_pixels_from_top
        cropped_col_height = half_height - crop_pixels_from_top

        if cropped_col_height <= 0:
            all_bubble_bboxes.append([(0, 0, 0, 0)] * options_per_question)
            continue

        option_slot_height = cropped_col_height // options_per_question

        question_options_bboxes = []
        for opt_idx in range(options_per_question):
            option_abs_y = cropped_col_abs_y + opt_idx * option_slot_height
            bbox = (col_abs_x, option_abs_y, column_width, option_slot_height)
            question_options_bboxes.append(bbox)
        all_bubble_bboxes.append(question_options_bboxes)

    bottom_half_abs_y = ans_sec_abs_y + half_height
    bottom_half_height = ans_sec_abs_h - half_height

    for col_idx in range(num_columns_per_half):
        col_abs_x = ans_sec_abs_x + col_idx * column_width

        crop_pixels_from_top = int(bottom_half_height * crop_percent_each_column)
        cropped_col_abs_y = bottom_half_abs_y + crop_pixels_from_top
        cropped_col_height = bottom_half_height - crop_pixels_from_top

        if cropped_col_height <= 0:
            all_bubble_bboxes.append([(0, 0, 0, 0)] * options_per_question)
            continue

        option_slot_height = cropped_col_height // options_per_question

        question_options_bboxes = []
        for opt_idx in range(options_per_question):
            option_abs_y = cropped_col_abs_y + opt_idx * option_slot_height
            bbox = (col_abs_x, option_abs_y, column_width, option_slot_height)
            question_options_bboxes.append(bbox)
        all_bubble_bboxes.append(question_options_bboxes)

    return all_bubble_bboxes


def find_and_draw_bubble_circle(image, bbox, color, thickness=3):
    """
    Finds a circle within the given bbox region and draws it on the original image.
    Adjust HoughCircles parameters as needed for your specific OMR.
    """
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return False

    roi = image[y:y + h, x:x + w]

    if roi.size == 0:
        return False

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Hough Circle Transform parameters - tune these!
    circles = cv2.HoughCircles(
        blurred_roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,        # Inverse ratio of the accumulator resolution
        minDist=max(w,h) // 2, # Minimum distance between centers (prevent duplicate detections)
        param1=100,    # Canny edge high threshold
        param2=20,     # Accumulator threshold (lower = more circles)
        minRadius=min(w,h) // 4,  # Min circle radius
        maxRadius=min(w,h) // 2  # Max circle radius (slightly less than half of the smaller dimension of the slot)
    )

    if circles is not None:
        circles = np.uint16(np.round(circles[0, :]))
        for i in circles[:1]: # Take only the first detected circle in the region
            center_x_rel, center_y_rel, radius = i[0], i[1], i[2]

            center_x_abs = x + center_x_rel
            center_y_abs = y + center_y_rel

            # Draw the circle with a slightly increased radius for visibility
            cv2.circle(image, (center_x_abs, center_y_abs), int(radius * 1.1), color, thickness)
            return True
    return False


# === Main execution block ===
if __name__ == "__main__":
    # --- TEACHER'S CORRECT ANSWERS (Keep static) ---
    correct_answers = [
        1, 2, 3, 4, 3, 2, 2, 3, 3, 4,
        3, 2, 2, 2, 2, 1, 2, 3, 3, 1,
        1, 1, 1, 2, 2, 2, 1, 2, 3, 4
    ]

    # --- Initialize Webcam ---
    # 0 usually refers to the default camera. Change if you have multiple webcams.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam. Please ensure it's connected and not in use.")
        exit()

    print("--- Starting Real-time OMR Sheet Processing (Press 'q' to quit) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to grab frame. Exiting...")
            break

        # Make a copy of the frame to draw on, keeping the original untouched for subsequent processing steps if needed
        original_image_for_display = frame.copy()

        # Step 1: Get the main OMR form (biggest rectangle)
        answer_section_img, info_section_img, info_section_abs_bbox, biggest_rect_overall = \
            display_and_split_biggest_rectangle_part(frame) # Pass the frame directly

        total_marks_calculated = 0
        student_id_bbox, subject_code_bbox, total_marks_bbox = None, None, None

        if biggest_rect_overall:
            # Step 2: Process the info section if available
            if info_section_img is not None and info_section_img.size > 0 and info_section_abs_bbox is not None:
                student_id_img, subject_code_img, total_marks_img, student_id_bbox, subject_code_bbox, total_marks_bbox = \
                    extract_info_sections(info_section_img, info_section_abs_bbox)

            # Step 3: Calculate pixel counts and determine answers
            all_column_black_pixel_counts = []
            all_determined_answers = []

            if answer_section_img is not None and answer_section_img.size > 0:
                PIXEL_SENSITIVITY_THRESHOLD = 100

                answer_top_half_cols, answer_bottom_half_cols = split_answer_section_horizontally(answer_section_img,
                                                                                                  crop_percent_each_column=0.18)

                all_bubble_bboxes = get_all_bubble_bboxes(original_image_for_display.shape, biggest_rect_overall,
                                                          crop_percent_each_column=0.18)

                # Process top half columns
                for col_img in answer_top_half_cols:
                    counts = split_and_count_black_pixels_in_parts(col_img, num_parts=4)
                    all_column_black_pixel_counts.append(counts)
                    determined_answer = get_answer_from_counts(counts, sensitivity_threshold=PIXEL_SENSITIVITY_THRESHOLD)
                    all_determined_answers.append(determined_answer)

                # Process bottom half columns
                for col_img in answer_bottom_half_cols:
                    counts = split_and_count_black_pixels_in_parts(col_img, num_parts=4)
                    all_column_black_pixel_counts.append(counts)
                    determined_answer = get_answer_from_counts(counts, sensitivity_threshold=PIXEL_SENSITIVITY_THRESHOLD)
                    all_determined_answers.append(determined_answer)

                # --- Mark Calculation and Highlighting ---
                if len(correct_answers) == len(all_determined_answers):
                    total_marks_calculated = 0
                    COLOR_GREEN = (0, 255, 0)  # Correct student answer
                    COLOR_RED = (0, 0, 255)  # Incorrect student answer or unmarked
                    COLOR_BLUE = (255, 0, 0)  # Correct answer when student's answer is wrong

                    for i in range(len(all_determined_answers)):
                        question_num = i + 1
                        student_ans = all_determined_answers[i]
                        correct_ans = correct_answers[i]

                        if i < len(all_bubble_bboxes) and len(all_bubble_bboxes[i]) == 4:
                            question_bubble_bboxes = all_bubble_bboxes[i]

                            if student_ans is not None:
                                if 1 <= student_ans <= 4:
                                    student_chosen_bbox = question_bubble_bboxes[student_ans - 1]
                                    if student_ans == correct_ans:
                                        total_marks_calculated += 1
                                        find_and_draw_bubble_circle(original_image_for_display, student_chosen_bbox, COLOR_GREEN)
                                    else:
                                        find_and_draw_bubble_circle(original_image_for_display, student_chosen_bbox, COLOR_RED)
                                        if 1 <= correct_ans <= 4:
                                            correct_bubble_bbox = question_bubble_bboxes[correct_ans - 1]
                                            find_and_draw_bubble_circle(original_image_for_display, correct_bubble_bbox, COLOR_BLUE)
                            else: # Unmarked or ambiguous by student
                                if 1 <= correct_ans <= 4:
                                    correct_bubble_bbox = question_bubble_bboxes[correct_ans - 1]
                                    find_and_draw_bubble_circle(original_image_for_display, correct_bubble_bbox, COLOR_RED)
                # --- Display Total Marks ---
                if total_marks_bbox is not None and original_image_for_display is not None:
                    text_to_display = str(total_marks_calculated)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_thickness = 3
                    font_color = (0, 0, 255)  # Red color (BGR)

                    box_x, box_y, box_w, box_h = total_marks_bbox
                    (text_width, text_height), baseline = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)
                    text_x = box_x + (box_w - text_width) // 2
                    text_y = box_y + (box_h + text_height) // 2 + (baseline // 2) - 10

                    cv2.putText(original_image_for_display, text_to_display, (text_x, text_y),
                                font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        else:
            # If no biggest rectangle is found, optionally display a message on the frame
            cv2.putText(original_image_for_display, "No OMR Form Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        # Display the result
        cv2.imshow("Real-time OMR Scanner", original_image_for_display)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Real-time Processing Stopped ---")