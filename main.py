import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from collections import Counter  # For frequency analysis


def detect_filled_circles(image_path, crop_top_percentage=0.18):
    """
    Detects filled circles in an image, useful for OMR sheet processing.
    The output image will also be cropped at the top.

    Args:
        image_path (str): The path to the input image.
        crop_top_percentage (float): The percentage of the image height to crop from the top.

    Returns:
        tuple: A tuple containing:
            - PIL.Image.Image: The CROPPED image with detected circles and status drawn on it.
            - int: The number of the selected option (1, 2, 3, 4).
                   Returns 0 if no option is selected, or if multiple options are detected.
    """
    try:
        # Load the image using OpenCV
        img_cv_original = cv2.imread(image_path)
        if img_cv_original is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Get original image dimensions
        height_original, width_original, _ = img_cv_original.shape

        # Calculate crop height
        crop_height = int(height_original * crop_top_percentage)

        # Crop the image for processing. This will be the base for our final output image.
        img_cv_processed_base = img_cv_original[crop_height:, 0:width_original]

        # Convert to grayscale for detection
        gray = cv2.cvtColor(img_cv_processed_base, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding.
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the thresholded image.
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Prepare a PIL image from the *cropped* OpenCV image to draw on
        img_pil = Image.fromarray(cv2.cvtColor(img_cv_processed_base, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Define font for drawing text
        try:
            # Using a system-wide font name or a path if needed
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            print("Could not load 'arial.ttf', using default PIL font. (Adjust font path if needed)")
            font = ImageFont.load_default()

        min_circle_area = 200  # Adjust this based on your image and circle size
        max_circle_area = 2000  # Adjust this based on your image and circle size
        min_circularity = 0.5  # A value close to 1 indicates a perfect circle

        potential_options = []

        # Loop over the found contours
        for i, c in enumerate(contours):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)

            if len(approx) > 5 and area > min_circle_area and area < max_circle_area:
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))  # These are already in the cropped image's coordinates
                radius = int(radius)

                if peri > 0:
                    circularity = (4 * np.pi * area) / (peri ** 2)
                else:
                    circularity = 0

                if circularity > min_circularity:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, center, radius - 2, (255), -1)

                    mean_intensity = cv2.mean(gray, mask=mask)[0]
                    is_filled = mean_intensity < 150  # Example threshold: adjust as needed

                    potential_options.append({
                        'center': center,  # Store coordinates relative to the cropped image
                        'radius': radius,
                        'is_filled': is_filled,
                        'mean_intensity': mean_intensity  # For debugging/tuning
                    })

        potential_options.sort(key=lambda opt: opt['center'][1])

        selected_option_numbers_from_this_image = []
        for option_num, opt in enumerate(potential_options, 1):
            x, y = opt['center']
            radius = opt['radius']
            is_filled = opt['is_filled']

            status_text = f"Option {option_num}: "
            if is_filled:
                draw_color = (0, 255, 0)  # Green for filled
                status_text += "Selected"
                selected_option_numbers_from_this_image.append(option_num)
            else:
                draw_color = (255, 0, 0)  # Red for empty
                status_text += "Not Selected"

            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=draw_color, width=2)
            draw.text((x + radius + 5, y), status_text, fill=draw_color, font=font)

        # Determine the single selected answer for this question
        if len(selected_option_numbers_from_this_image) == 1:
            return img_pil, selected_option_numbers_from_this_image[0]
        elif len(selected_option_numbers_from_this_image) > 1:
            # More than one option selected for a single question -> consider it incorrect/ambiguous
            print(
                f"  Warning for {os.path.basename(image_path)}: Multiple options selected ({selected_option_numbers_from_this_image}). Counted as 0 marks.")
            return img_pil, 0
        else:
            # No option selected
            return img_pil, 0

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return None, 0  # Return 0 for the answer if an error occurs


if __name__ == '__main__':
    answer_columns_dir = 'answer_columns'
    output_processed_dir = 'processed_omr_sheets_cropped_output'
    os.makedirs(output_processed_dir, exist_ok=True)

    # --- Define the correct answers here ---
    corrected_answers = [1, 2, 3, 4, 3, 2, 2, 3, 3, 4, 3, 2, 2, 2, 2, 1, 2, 3, 3, 1, 1, 1, 1, 2, 2, 2, 1, 2, 3, 4]

    student_answers = []  # To store the detected answers for each question in order

    # --- Adjust this value based on your images ---
    # Increase if the question number header is still visible.
    # Decrease if actual answer circles are being cut off.
    CROP_PERCENTAGE = 0.18  # Example: Crop top 18%

    # Get a sorted list of image filenames based on the question number
    # This is crucial for matching answers with the corrected_answers list
    image_files_to_process = sorted([f for f in os.listdir(answer_columns_dir)
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))],
                                    key=lambda x: int(x.split('_')[1].split('.')[
                                                          0]) if '_' in x else 0)  # Handles 'question_X.png' format

    print(f"Total image files to process: {len(image_files_to_process)}")
    print(f"Total correct answers defined: {len(corrected_answers)}")

    if len(image_files_to_process) != len(corrected_answers):
        print("\nWARNING: The number of image files does not match the number of correct answers.")
        print("  This might lead to incorrect scoring. Please ensure one image per question.")
        # Proceed anyway, but the score calculation might be off.

    for filename in image_files_to_process:
        image_path = os.path.join(answer_columns_dir, filename)

        print(f"\n--- Processing: {filename} (Output will be cropped by {CROP_PERCENTAGE * 100:.0f}%) ---")

        # Call the detection function. It now returns a single answer (or 0).
        processed_image, selected_answer_for_question = detect_filled_circles(image_path,
                                                                              crop_top_percentage=CROP_PERCENTAGE)

        if processed_image:
            output_path = os.path.join(output_processed_dir, f"processed_cropped_output_{filename}")
            processed_image.save(output_path)
            print(f"Processed image saved to {output_path}")
            print(f"Selected option for {filename}: {selected_answer_for_question}")
            student_answers.append(selected_answer_for_question)
        else:
            print(f"Skipping {filename} due to an error during processing. Assigning 0 for this question.")
            student_answers.append(0)  # Assign 0 if an error occurred during image processing

    print("\n--- All Images Processed ---")
    print(f"Student's detected answers: {student_answers}")
    print(f"Correct answers:           {corrected_answers}")
    print(f"Total questions processed: {len(student_answers)}")

    # --- Calculate Marks ---
    marks_obtained = 0
    if len(student_answers) == len(corrected_answers):
        for i in range(len(corrected_answers)):
            if student_answers[i] == corrected_answers[i]:
                marks_obtained += 1
        print(f"\nMarks obtained: {marks_obtained} out of {len(corrected_answers)}")
    else:
        print(
            "\nCannot calculate exact marks: Number of detected answers does not match the number of correct answers.")
        print(f"Detected answers count: {len(student_answers)}, Correct answers count: {len(corrected_answers)}")
        # You could add logic here to calculate a partial score if desired

    # Optional: Further analysis of detected answers
    answer_counts = Counter(student_answers)
    print(f"Frequency of each detected option (0 means unselected/multiple/error): {answer_counts}")