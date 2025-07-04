import cv2
import numpy as np
import os


def display_and_split_biggest_rectangle_part(image_path):
    """
    Loads an image, finds the biggest rectangular part, and splits it into
    the answer section and info section, returning them as NumPy arrays.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None, None  # Return None for both parts on failure

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_rect = None
    max_area = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area > max_area:
                max_area = area
                biggest_rect = (x, y, w, h)

    if biggest_rect:
        x, y, w, h = biggest_rect
        rectangular_part = image[y:y + h, x:x + w]

        # Optional: Display the biggest rectangle
        # cv2.imshow("Biggest Rectangle Found", rectangular_part)
        # cv2.waitKey(0)

        # Split into initial answer section and info section
        # Assuming the answer section is on the left (75% width) and info on the right (25% width)
        answer_section_with_outer = rectangular_part[:, 0:int(0.75 * w)]
        info_section_with_outer = rectangular_part[:, int(0.75 * w):]

        # Function to find the main content within a section (removing white borders)
        def crop_white_borders(section_image):
            if section_image is None or section_image.size == 0:
                return section_image  # Return empty if input is empty
            gray_section = cv2.cvtColor(section_image, cv2.COLOR_BGR2GRAY)
            # Adjust threshold if needed for better results
            _, thresh_section = cv2.threshold(gray_section, 200, 255, cv2.THRESH_BINARY_INV)
            contours_section, _ = cv2.findContours(thresh_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours_section:
                largest_contour_section = max(contours_section, key=cv2.contourArea)
                x_sec, y_sec, w_sec, h_sec = cv2.boundingRect(largest_contour_section)
                return section_image[y_sec:y_sec + h_sec, x_sec:x_sec + w_sec]
            return section_image  # Return original if no contours found

        answer_section_without_outer = crop_white_borders(answer_section_with_outer)
        info_section_without_outer = crop_white_borders(info_section_with_outer)

        print("✅ Main sections extracted (in memory).")

        return answer_section_without_outer, info_section_without_outer
    else:
        print("❌ No significant rectangle found. Try adjusting the threshold or contour area.")
        return None, None  # Return None for both parts on failure


def extract_info_sections(info_section_image):
    """
    Extracts the 'শিক্ষার্থী আইডি', 'বিষয় কোড', and 'মোট নম্বর' sections
    from the info_section_image (NumPy array) and returns them as NumPy arrays.
    """
    if info_section_image is None or info_section_image.size == 0:
        print("❌ Error: Info section image is empty or None.")
        return None, None, None

    img_height, img_width, _ = info_section_image.shape

    # --- ADJUST THESE PERCENTAGES ---
    student_id_section_height_percentage = 0.25
    subject_code_section_height_percentage = 0.25

    student_id_start_y = 0
    student_id_end_y = int(img_height * student_id_section_height_percentage)

    subject_code_start_y = student_id_end_y
    subject_code_end_y = subject_code_start_y + int(img_height * subject_code_section_height_percentage)

    total_marks_start_y = subject_code_end_y
    total_marks_end_y = img_height

    student_id_part = None
    subject_code_part = None
    total_marks_part = None

    if student_id_end_y > student_id_start_y:
        student_id_part = info_section_image[student_id_start_y:student_id_end_y, 0:img_width]
        print("✅ Student ID section extracted (in memory).")
    else:
        print("❌ Student ID section crop failed.")

    if subject_code_end_y > subject_code_start_y:
        subject_code_part = info_section_image[subject_code_start_y:subject_code_end_y, 0:img_width]
        print("✅ Subject Code section extracted (in memory).")
    else:
        print("❌ Subject Code section crop failed.")

    if total_marks_end_y > total_marks_start_y:
        total_marks_part = info_section_image[total_marks_start_y:total_marks_end_y, 0:img_width]
        print("✅ Total Marks section extracted (in memory).")
    else:
        print("❌ Total Marks section crop failed.")

    return student_id_part, subject_code_part, total_marks_part


def crop_top_percentage(image, percentage):
    """
    Crops the top 'percentage' of an image.
    """
    if image is None or image.size == 0:
        return image

    height, width, _ = image.shape
    crop_height = int(height * percentage)

    # Ensure crop_height doesn't exceed image height
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
        if cropped_col.size > 0:  # Add only if not empty after crop
            cropped_top_half_cols.append(cropped_col)
        else:
            print(f"⚠️ Warning: Top Half Col {i + 1} became empty after {crop_percent_each_column * 100}% top crop.")

    # Split bottom_half into 15 columns and then crop each
    bottom_half_cols = np.array_split(bottom_half, num_columns, axis=1)
    print(f"✅ Bottom half split into {len(bottom_half_cols)} columns.")
    for i, col_img in enumerate(bottom_half_cols):
        cropped_col = crop_top_percentage(col_img, crop_percent_each_column)
        if cropped_col.size > 0:  # Add only if not empty after crop
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

    # Convert to grayscale for black pixel counting
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
            # Count pixels where intensity is below the black_pixel_threshold
            # Adjust this threshold based on what you consider "black" in your images
            black_pixels = np.sum(part < black_pixel_threshold)
            black_pixel_counts.append(int(black_pixels))  # Ensure integer
        else:
            black_pixel_counts.append(0)  # No pixels if part is empty

    return black_pixel_counts


def get_answer_from_counts(pixel_counts, sensitivity_threshold=20):
    """
    Determines the selected answer (1-4) based on black pixel counts.
    Assumes the highest count indicates the answer.
    'sensitivity_threshold' is the minimum pixel count to consider a bubble filled.
    """
    if not pixel_counts:
        return None  # No counts provided

    max_pixels = 0
    selected_answer_index = -1

    for i, count in enumerate(pixel_counts):
        if count > max_pixels and count >= sensitivity_threshold:
            max_pixels = count
            selected_answer_index = i

    if selected_answer_index != -1:
        return selected_answer_index + 1  # Convert 0-indexed to 1-indexed answer
    else:
        # If no count crosses the sensitivity_threshold, it means no clear answer
        # If you want to return 0 for unmarked answers, change None to 0
        return None  # Or 'No Answer', 'Ambiguous', etc.


# === Main execution block ===
if __name__ == "__main__":
    original_omr_image_path = "omr5.png"  # Your original OMR image file

    print("--- Starting OMR Sheet Processing ---")

    # Step 1: Get the main answer and info sections
    answer_section_img, info_section_img = display_and_split_biggest_rectangle_part(original_omr_image_path)

    # Step 2: Process the info section if available
    if info_section_img is not None and info_section_img.size > 0:
        print("\n--- Extracting Info Sub-sections ---")
        student_id_img, subject_code_img, total_marks_img = extract_info_sections(info_section_img)

        # Optional: Display info sub-sections (no saving)
        if student_id_img is not None: cv2.imshow("Student ID", student_id_img)
        if subject_code_img is not None: cv2.imshow("Subject Code", subject_code_img)
        if total_marks_img is not None: cv2.imshow("Total Marks", total_marks_img)
        cv2.waitKey(1)  # Small wait key to allow windows to pop up

    # Step 3: Split the answer section if available, crop top 18%, and then divide and count black pixels
    if answer_section_img is not None and answer_section_img.size > 0:
        print("\n--- Splitting Answer Section into Halves, Cropping Top 18%, and Analyzing Columns ---")
        # Adjust sensitivity_threshold=100 based on your OMR image's filled bubble pixel counts
        # This threshold determines what minimum black pixel count is considered a "filled" bubble.
        PIXEL_SENSITIVITY_THRESHOLD = 100

        answer_top_half_cols, answer_bottom_half_cols = split_answer_section_horizontally(answer_section_img,
                                                                                          crop_percent_each_column=0.18)

        all_column_black_pixel_counts = []
        all_determined_answers = []

        # Process top half columns
        if answer_top_half_cols:
            print("\nAnalyzing Answer Top Half Cropped Columns:")
            for i, col_img in enumerate(answer_top_half_cols):
                counts = split_and_count_black_pixels_in_parts(col_img, num_parts=4)
                all_column_black_pixel_counts.append(counts)
                determined_answer = get_answer_from_counts(counts, sensitivity_threshold=PIXEL_SENSITIVITY_THRESHOLD)
                all_determined_answers.append(determined_answer)
                print(f"  Top Half Col {i + 1} Black Pixel Counts (4 parts): {counts} -> Answer: {determined_answer}")
                cv2.imshow(f"Top Half Col {i + 1} (Cropped)", col_img)

                # Process bottom half columns
        if answer_bottom_half_cols:
            print("\nAnalyzing Answer Bottom Half Cropped Columns:")
            for i, col_img in enumerate(answer_bottom_half_cols):
                counts = split_and_count_black_pixels_in_parts(col_img, num_parts=4)
                all_column_black_pixel_counts.append(counts)
                # Note: We continue column numbering from 16 for bottom half for display
                determined_answer = get_answer_from_counts(counts, sensitivity_threshold=PIXEL_SENSITIVITY_THRESHOLD)
                all_determined_answers.append(determined_answer)
                print(
                    f"  Bottom Half Col {i + 1 + 15} Black Pixel Counts (4 parts): {counts} -> Answer: {determined_answer}")
                cv2.imshow(f"Bottom Half Col {i + 1 + 15} (Cropped)", col_img)

        print("\n--- Answer Evaluation ---")
        print("Summary of all column black pixel counts:")
        for i, counts in enumerate(all_column_black_pixel_counts):
            print(f"Column {i + 1}: {counts}")

        print("\nDetermined Answers for all 30 Questions:")
        for i, answer in enumerate(all_determined_answers):
            print(f"Question {i + 1}: {answer if answer is not None else 'Unmarked/Ambiguous'}")

        # --- TEACHER'S CORRECT ANSWERS ---
        # IMPORTANT: Populate this array with the actual correct answers (1-4) for your 30 questions.
        # Example: if Q1=1, Q2=2, Q3=3, Q4=4, Q5=3, Q6=2, Q7=2, ...
        # Make sure this list has exactly 30 answers.
        correct_answers = [
            1, 2, 3, 4, 3, 2, 2, 3, 3, 4,
            3, 2, 2, 2, 2, 1, 2, 3, 3, 1,
            1, 1, 1, 2, 2, 2, 1, 2, 3, 4
        ]
        # Based on your provided pixel counts, the first 15 answers are:
        # [1, 2, 3, 4, 3, 2, 2, 3, 3, 4, 3, 2, 2, 2, 2]
        # And for the bottom 15 (questions 16-30):
        # [1, 2, 3, 3, 1, 1, 1, 1, 2, 2, 2, 1, 2, 3, 4]
        # I've updated the example `correct_answers` to match the answers your example output implies.

        if len(correct_answers) != len(all_determined_answers):
            print(
                f"\n❌ Error: Mismatch in number of correct answers ({len(correct_answers)}) and determined answers ({len(all_determined_answers)}). Cannot calculate marks.")
        else:
            total_marks = 0
            detailed_results = []
            for i in range(len(all_determined_answers)):
                question_num = i + 1
                student_ans = all_determined_answers[i]
                correct_ans = correct_answers[i]

                is_correct = False
                if student_ans is not None and student_ans == correct_ans:
                    total_marks += 1
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

            print(f"\n✅ Total Marks: {total_marks} / {len(correct_answers)}")

        cv2.waitKey(0)  # Wait for a key press to close all column windows
        cv2.destroyAllWindows()
    else:
        print("\nAnswer section not available for further analysis or mark calculation.")

    print("\n--- Processing Complete ---")