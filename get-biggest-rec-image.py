import cv2
import numpy as np
import os

def split_info_section_into_labeled_boxes(info_section_image_path):
    """
    Splits the info section image (containing Student ID, Subject Code, Total Number)
    into separate images for each labeled section and its input area.
    """
    info_image = cv2.imread(info_section_image_path)

    if info_image is None:
        print(f"❌ Error: Could not load info section image from {info_section_image_path}")
        return

    height, width, _ = info_image.shape
    gray = cv2.cvtColor(info_image, cv2.COLOR_BGR2GRAY)

    # Convert to binary image (threshold) - crucial for finding distinct regions
    # Adjust this threshold if the pink background or text is not clearly separating
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV) # High threshold to isolate dark text/lines on light background

    # Find horizontal projection profile to detect clear separations
    horizontal_proj = np.sum(binary, axis=1)

    # Find the 'valleys' or 'low points' in the projection, which indicate
    # the separation lines between the sections.
    # This part might need fine-tuning based on the actual pixel values
    # and consistency of your info_section.png
    separators_y = []
    threshold_for_separation = 100 # A low sum of pixels could indicate a gap
    for y in range(1, height - 1):
        # Look for a sharp drop or a region with very few non-zero pixels
        # indicating a blank horizontal line or the thin white lines between pink sections
        if horizontal_proj[y] < threshold_for_separation and horizontal_proj[y-1] >= threshold_for_separation:
             separators_y.append(y)
        elif horizontal_proj[y] >= threshold_for_separation and horizontal_proj[y+1] < threshold_for_separation:
             separators_y.append(y+1)


    # Filter and refine separators_y to get the main dividers
    # The actual points where your pink sections end and new ones begin, or the white lines
    # Based on your image, there are lines around:
    # 1. After "শিক্ষার্থী আইডি" box
    # 2. After "বিষয় কোড" box
    # 3. After "মোট নম্বর" label (before the large input area)

    # A more robust way might be to look for changes in color or strong edge detection.
    # Let's try a simpler approach of fixed ratios or direct segmentation by known height
    # if the sections are always consistently sized.

    # Given the fixed structure of your image:
    # Top section: "শিক্ষার্থী আইডি" + its input box
    # Middle section: "বিষয় কোড" + its input box
    # Bottom section: "মোট নম্বর" + its input area

    # Estimate section boundaries based on the visual layout (approximate y-coordinates)
    # These values are eyeballed from your image and might need adjustment.
    # This assumes info_section.png is cropped consistently.
    section_boundaries = [
        0, # Start of the first section
        height // 3, # End of first section, start of second
        (2 * height) // 3, # End of second section, start of third
        height # End of the image
    ]

    # Refine boundaries by finding the actual distinct horizontal lines
    # (e.g., the white lines between the pink blocks or the bottom of the input boxes)
    # This requires more advanced image processing to robustly detect these specific lines.
    # For now, let's use a simpler fixed division or rely on detected significant horizontal lines.

    # Let's re-try finding strong horizontal lines (could be the white separators)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=width*0.8, maxLineGap=10)

    horizontal_line_y_coords = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 5: # It's a horizontal line if y coordinates are close
                horizontal_line_y_coords.append(y1)

    # Sort and unique the y-coordinates
    horizontal_line_y_coords = sorted(list(set(horizontal_line_y_coords)))

    # Filter out lines that are too close to each other or at the very top/bottom
    filtered_lines_y = []
    if horizontal_line_y_coords:
        filtered_lines_y.append(horizontal_line_y_coords[0])
        for i in range(1, len(horizontal_line_y_coords)):
            if horizontal_line_y_coords[i] - filtered_lines_y[-1] > 20: # Ensure lines are sufficiently separated
                filtered_lines_y.append(horizontal_line_y_coords[i])

    # Add the top and bottom of the image as implicit boundaries
    final_boundaries = [0] + filtered_lines_y + [height]
    final_boundaries = sorted(list(set(final_boundaries)))

    # Ensure we have at least 4 boundaries for 3 sections (0, y1, y2, y3, height)
    if len(final_boundaries) < 4:
        print("Warning: Not enough clear horizontal separation lines found. Using approximate division.")
        # Fallback to approximate division if lines are not detected
        final_boundaries = [0, height // 3, (2 * height) // 3, height]
        # Further refine by trying to find the blank input box regions if needed

    section_names = ["student_id_section", "subject_code_section", "total_number_section"]
    saved_files = []

    for i in range(len(final_boundaries) - 1):
        if i < len(section_names):
            y_start = final_boundaries[i]
            y_end = final_boundaries[i+1]
            if y_end > y_start: # Ensure valid section
                section_part = info_image[y_start:y_end, 0:width]
                output_filename = f"{section_names[i]}.png"
                cv2.imwrite(output_filename, section_part)
                saved_files.append(output_filename)
                print(f"✅ Saved: {output_filename}")
            else:
                print(f"Skipping empty section at boundaries {y_start}-{y_end}")


    if saved_files:
        print(f"Successfully extracted {len(saved_files)} individual sections from info_section.png.")
        # Optional: Display them
        # for filename in saved_files:
        #     cv2.imshow(filename, cv2.imread(filename))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("❌ No individual sections found within info_section.png. Adjust parameters or input image.")


# --- Main call ---
if __name__ == "__main__":
    # Make sure 'info_section.png' is generated by your first script part or exists here.
    # If it doesn't exist, you need to run the initial part of your code first to create it.
    info_section_file = "info_section.png"
    if os.path.exists(info_section_file):
        split_info_section_into_labeled_boxes(info_section_file)
    else:
        print(f"Error: {info_section_file} not found. Please ensure it's generated first.")