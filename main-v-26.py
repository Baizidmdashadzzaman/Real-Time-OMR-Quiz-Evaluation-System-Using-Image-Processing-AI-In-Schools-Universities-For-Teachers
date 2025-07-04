import cv2
import numpy as np
import os


def display_and_split_biggest_rectangle_part(image_path):
    """
    Loads an image, finds the biggest rectangular part, and splits it into
    the answer section and info section. It returns these as NumPy arrays
    along with the absolute bounding box of the info section (after initial cropping).
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None, None, None  # Return None for all parts on failure

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

    info_section_abs_bbox = None  # Initialize to None

    if biggest_rect:
        x, y, w, h = biggest_rect
        rectangular_part = image[y:y + h, x:x + w]

        # Calculate coordinates for info section within the original image
        info_section_with_outer_abs_x = x + int(0.75 * w)
        info_section_with_outer_abs_y = y
        # The width of the info section is the remaining 25% of the biggest rect's width
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
                return section_image, (current_abs_x, current_abs_y, 0, 0)  # Return empty and 0 bbox

            gray_section = cv2.cvtColor(section_image, cv2.COLOR_BGR2GRAY)
            _, thresh_section = cv2.threshold(gray_section, 200, 255, cv2.THRESH_BINARY_INV)
            contours_section, _ = cv2.findContours(thresh_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_section:
                # Find the largest contour within this section
                largest_contour_section = max(contours_section, key=cv2.contourArea)
                x_sec, y_sec, w_sec, h_sec = cv2.boundingRect(largest_contour_section)

                # Calculate absolute coordinates for the cropped section
                abs_x = current_abs_x + x_sec
                abs_y = current_abs_y + y_sec
                abs_w = w_sec
                abs_h = h_sec

                return section_image[y_sec:y_sec + h_sec, x_sec:x_sec + w_sec], (abs_x, abs_y, abs_w, abs_h)

            # If no contours found, return original section and its assumed absolute bbox
            return section_image, (current_abs_x, current_abs_y, section_image.shape[1], section_image.shape[0])

        # We don't need the bbox for the answer section for this task, so discard it
        answer_section_without_outer, _ = crop_white_borders_with_offset(answer_section_with_outer, 0, 0)

        # Get the info section cropped and its absolute bbox
        info_section_without_outer, info_section_abs_bbox = crop_white_borders_with_offset(
            info_section_with_outer, info_section_with_outer_abs_x, info_section_with_outer_abs_y
        )

        print("✅ Main sections extracted (in memory).")
        return answer_section_without_outer, info_section_without_outer, info_section_abs_bbox
    else:
        print("❌ No significant rectangle found. Try adjusting the threshold or contour area.")
        return None, None, None  # Return None for all parts on failure


def extract_info_sections(info_section_image, info_section_abs_bbox):
    """
    Extracts the 'শিক্ষার্থী আইডি', 'বিষয় কোড', and 'মোট নম্বর' sections
    from the info_section_image (NumPy array) and returns them as NumPy arrays
    along with their absolute bounding boxes in the original image.
    """
    if info_section_image is None or info_section_image.size == 0 or info_section_abs_bbox is None:
        print("❌ Error: Info section image or its bbox is empty or None.")
        return None, None, None, None, None, None  # Return None for all parts on failure

    img_height, img_width, _ = info_section_image.shape

    info_abs_x, info_abs_y, _, _ = info_section_abs_bbox  # Get absolute x,y of the info section

    # --- ADJUST THESE PERCENTAGES (relative to the info_section_image height) ---
    student_id_section_height_percentage = 0.25
    subject_code_section_height_percentage = 0.25

    student_id_start_y_rel = 0
    student_id_end_y_rel = int(img_height * student_id_section_height_percentage)

    subject_code_start_y_rel = student_id_end_y_rel
    subject_code_end_y_rel = subject_code_start_y_rel + int(img_height * subject_code_section_height_percentage)

    total_marks_start_y_rel = subject_code_end_y_rel
    total_marks_end_y_rel = img_height  # Ensure it goes to the end of the image

    student_id_part = None
    subject_code_part = None
    total_marks_part = None
    student_id_bbox = None
    subject_code_bbox = None
    total_marks_bbox = None

    if student_id_end_y_rel > student_id_start_y_rel:
        student_id_part = info_section_image[student_id_start_y_rel:student_id_end_y_rel, 0:img_width]
        # Calculate absolute bounding box for student ID part
        student_id_bbox = (info_abs_x, info_abs_y + student_id_start_y_rel, img_width,
                           student_id_end_y_rel - student_id_start_y_rel)
        print("✅ Student ID section extracted (in memory).")
    else:
        print("❌ Student ID section crop failed.")

    if subject_code_end_y_rel > subject_code_start_y_rel:
        subject_code_part = info_section_image[subject_code_start_y_rel:subject_code_end_y_rel, 0:img_width]
        # Calculate absolute bounding box for subject code part
        subject_code_bbox = (info_abs_x, info_abs_y + subject_code_start_y_rel, img_width,
                             subject_code_end_y_rel - subject_code_start_y_rel)
        print("✅ Subject Code section extracted (in memory).")
    else:
        print("❌ Subject Code section crop failed.")

    if total_marks_end_y_rel > total_marks_start_y_rel:
        total_marks_part = info_section_image[total_marks_start_y_rel:total_marks_end_y_rel, 0:img_width]
        # Calculate absolute bounding box for total marks part
        total_marks_bbox = (info_abs_x, info_abs_y + total_marks_start_y_rel, img_width,
                            total_marks_end_y_rel - total_marks_start_y_rel)
        print("✅ Total Marks section extracted (in memory).")
    else:
        print("❌ Total Marks section crop failed.")

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
        print(
            f"⚠️ Warning: Crop height ({crop_height}) is greater than or equal to image height ({height}). Returning empty array.")
        return np.array([])  # Return empty if cropping would remove all content or more

    return image[crop_height:height, 0:width]


def split_answer_section_horizontally(answer_section_image, crop_percent_each_column=0.18):
    """
    Splits the answer_section_image (NumPy array) horizontally into two equal halves,
    and then each half into 15 columns. Then, crops the top 'crop_percent_each_column'
    from each of these 30 columns. Returns all these parts as a list of NumPy arrays.
    """
    if answer_section_image is None or answer_section_image.size == 0:
        print(f"❌ Error: Answer section image is empty or None.")
        return [], []  # Return empty lists on failure

    img_height, img_width, _ = answer_section_image.shape

    middle_y = img_height // 2

    top_half = answer_section_image[0:middle_y, 0:img_width]
    bottom_half = answer_section_image[middle_y:img_height, 0:img_width]

    print("✅ Answer section split into two halves (in memory).")

    num_columns = 15
    cropped_top_half_cols = []
    cropped_bottom_half_cols = []

    # Split top_half into 15 columns and then crop each
    top_half_cols = np.array_split(top_half, num_columns, axis=1)
    print(f"✅ Top half split into {len(top_half_cols)} columns.")
    for i, col_img in enumerate(top_half_cols):
        cropped_col = crop_top_percentage(col_img, crop_percent_each_column)
        if cropped_col.size > 0:
            cropped_top_half_cols.append(cropped_col)
        else:
            print(f"⚠️ Warning: Top Half Col {i + 1} became empty after {crop_percent_each_column * 100}% top crop.")

    # Split bottom_half into 15 columns and then crop each
    bottom_half_cols = np.array_split(bottom_half, num_columns, axis=1)
    print(f"✅ Bottom half split into {len(bottom_half_cols)} columns.")
    for i, col_img in enumerate(bottom_half_cols):
        cropped_col = crop_top_percentage(col_img, crop_percent_each_column)
        if cropped_col.size > 0:
            cropped_bottom_half_cols.append(cropped_col)
        else:
            print(f"⚠️ Warning: Bottom Half Col {i + 1} became empty after {crop_percent_each_column * 100}% top crop.")

    return cropped_top_half_cols, cropped_bottom_half_cols


def split_and_count_black_pixels_in_parts(column_image, num_parts=4, black_pixel_threshold=50):
    """
    Splits a single column image horizontally into 'num_parts' equal parts
    and counts the number of 'black' pixels in each part.
    'black_pixel_threshold' defines what intensity value is considered black (0-255).
    """
    if column_image is None or column_image.size == 0:
        print("❌ Error: Empty column image provided for splitting and counting.")
        return []

    gray_column = cv2.cvtColor(column_image, cv2.COLOR_BGR2GRAY)

    height, width = gray_column.shape
    part_height = height // num_parts
    black_pixel_counts = []

    for i in range(num_parts):
        start_y = i * part_height
        end_y = (i + 1) * part_height

        # Ensure the last part takes any remaining pixels due to integer division
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

    for i, count in enumerate(pixel_counts):
        if count > max_pixels and count >= sensitivity_threshold:
            max_pixels = count
            selected_answer_index = i

    if selected_answer_index != -1:
        return selected_answer_index + 1  # Convert 0-indexed to 1-indexed answer
    else:
        return None  # Or 'Unmarked'


# === Main execution block ===
if __name__ == "__main__":
    original_omr_image_path = "omr5.png"  # Your original OMR image file

    print("--- Starting OMR Sheet Processing ---")

    # Load the original image once to draw on it later
    original_image_for_display = cv2.imread(original_omr_image_path)
    if original_image_for_display is None:
        print(f"❌ Error: Could not load original image from {original_omr_image_path} for display. Exiting.")
        exit()

        # Step 1: Get the main answer and info sections (now also returns info_section_abs_bbox)
    answer_section_img, info_section_img, info_section_abs_bbox = display_and_split_biggest_rectangle_part(
        original_omr_image_path)

    # Step 2: Process the info section if available
    student_id_bbox, subject_code_bbox, total_marks_bbox = None, None, None
    if info_section_img is not None and info_section_img.size > 0 and info_section_abs_bbox is not None:
        print("\n--- Extracting Info Sub-sections ---")
        student_id_img, subject_code_img, total_marks_img, student_id_bbox, subject_code_bbox, total_marks_bbox = \
            extract_info_sections(info_section_img, info_section_abs_bbox)

        # Optional: Display info sub-sections for debugging (remove or comment out for production)
        # if student_id_img is not None: cv2.imshow("Student ID", student_id_img)
        # if subject_code_img is not None: cv2.imshow("Subject Code", subject_code_img)
        # if total_marks_img is not None: cv2.imshow("Total Marks", total_marks_img)
        # cv2.waitKey(1) # Small wait key to allow windows to pop up

    # Step 3: Split the answer section, analyze, and calculate marks
    total_marks_calculated = 0  # Initialize total marks

    if answer_section_img is not None and answer_section_img.size > 0:
        print("\n--- Analyzing Answer Columns ---")
        # Adjust sensitivity_threshold based on your OMR image's filled bubble pixel counts
        # This threshold determines what minimum black pixel count is considered a "filled" bubble.
        PIXEL_SENSITIVITY_THRESHOLD = 100

        answer_top_half_cols, answer_bottom_half_cols = split_answer_section_horizontally(answer_section_img,
                                                                                          crop_percent_each_column=0.18)

        all_column_black_pixel_counts = []
        all_determined_answers = []

        # Process top half columns
        if answer_top_half_cols:
            for i, col_img in enumerate(answer_top_half_cols):
                counts = split_and_count_black_pixels_in_parts(col_img, num_parts=4)
                all_column_black_pixel_counts.append(counts)
                determined_answer = get_answer_from_counts(counts, sensitivity_threshold=PIXEL_SENSITIVITY_THRESHOLD)
                all_determined_answers.append(determined_answer)

        # Process bottom half columns
        if answer_bottom_half_cols:
            for i, col_img in enumerate(answer_bottom_half_cols):
                counts = split_and_count_black_pixels_in_parts(col_img, num_parts=4)
                all_column_black_pixel_counts.append(counts)
                determined_answer = get_answer_from_counts(counts, sensitivity_threshold=PIXEL_SENSITIVITY_THRESHOLD)
                all_determined_answers.append(determined_answer)

        print("\n--- Answer Evaluation ---")
        print("Summary of all column black pixel counts:")
        for i, counts in enumerate(all_column_black_pixel_counts):
            print(f"Column {i + 1}: {counts}")

        print("\nDetermined Answers for all 30 Questions:")
        for i, answer in enumerate(all_determined_answers):
            print(f"Question {i + 1}: {answer if answer is not None else 'Unmarked/Ambiguous'}")

        # --- TEACHER'S CORRECT ANSWERS ---
        # IMPORTANT: Populate this array with the actual correct answers (1-4) for your 30 questions.
        # Ensure this list has exactly 30 answers.
        correct_answers = [
            1, 2, 3, 4, 3, 2, 2, 3, 3, 4,
            3, 2, 2, 2, 2, 1, 2, 3, 3, 1,
            1, 1, 1, 2, 2, 2, 1, 2, 3, 4
        ]
        # These example answers match the answers implied by your previous sample pixel counts.

        if len(correct_answers) != len(all_determined_answers):
            print(
                f"\n❌ Error: Mismatch in number of correct answers ({len(correct_answers)}) and determined answers ({len(all_determined_answers)}). Cannot calculate marks.")
        else:
            total_marks_calculated = 0
            detailed_results = []
            for i in range(len(all_determined_answers)):
                question_num = i + 1
                student_ans = all_determined_answers[i]
                correct_ans = correct_answers[i]

                is_correct = False
                if student_ans is not None and student_ans == correct_ans:
                    total_marks_calculated += 1
                    is_correct = True

                detailed_results.append({
                    "Question": question_num,
                    "Student Answer": student_ans if student_ans is not None else "Unmarked",
                    "Correct Answer": correct_ans,
                    "Status": "Correct" if is_correct else "Incorrect" if student_ans is not None else "Unmarked"
                })

            print("\n--- Mark Calculation Details ---")
            for result in detailed_results:
                print(
                    f"Q{result['Question']}: Student: {result['Student Answer']}, Correct: {result['Correct Answer']}, Status: {result['Status']}")

            print(f"\n✅ Total Marks: {total_marks_calculated} / {len(correct_answers)}")

        # --- Display Total Marks on omr4.png in the 'Total Marks' section ---
        if total_marks_bbox is not None and original_image_for_display is not None:
            text_to_display = str(total_marks_calculated)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # Adjust as needed for size
            font_thickness = 3  # Adjust as needed for boldness
            font_color = (0, 0, 255)  # Red color (BGR) for visibility

            # Get the coordinates and dimensions of the total_marks_bbox
            box_x, box_y, box_w, box_h = total_marks_bbox

            # Calculate text size to help with centering
            (text_width, text_height), baseline = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)

            # Position the text: center horizontally, and adjust vertically to fit within the box.
            # Using baseline for text_y often works well for bottom-aligned text, then adjust upwards.
            text_x = box_x + (box_w - text_width) // 2
            text_y = box_y + (box_h + text_height) // 2 + (baseline // 2) - 10  # Adjust -10 to move up/down slightly

            # Draw the text on the original image
            cv2.putText(original_image_for_display, text_to_display, (text_x, text_y),
                        font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            print(
                f"\n✅ Displaying total marks ({total_marks_calculated}) on {original_omr_image_path} in 'Total Marks' section.")
            cv2.imshow("OMR with Total Marks", original_image_for_display)
            cv2.waitKey(0)  # Wait for a key press to close the final image
            cv2.destroyAllWindows()
        else:
            print(
                "❌ Could not display total marks on image: Total Marks section bounding box or original image not available.")
            # If we couldn't display the marks on the image, ensure all other windows are closed
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("\nAnswer section not available for further analysis or mark calculation.")
        # If no answer section was found, close any other windows that might be open
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n--- Processing Complete ---")