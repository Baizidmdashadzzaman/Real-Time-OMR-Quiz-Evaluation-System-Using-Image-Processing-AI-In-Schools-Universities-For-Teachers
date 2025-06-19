import cv2
import numpy as np


def find_biggest_rectangle_robust(image_or_roi, min_contour_area=1000):
    """
    Finds the biggest rectangular contour in a given image or Region of Interest (ROI)
    with improved robustness for slightly broken lines using morphological operations.

    Args:
        image_or_roi (np.array): The input image (BGR) or ROI from which to find the rectangle.
        min_contour_area (int): Minimum area for a contour to be considered.

    Returns:
        tuple: A tuple containing (cropped_rectangle_image, (x, y, w, h)) if found,
               otherwise (None, None).
    """
    if image_or_roi is None:
        print("Input image or ROI is None.")
        return None, None

    gray = cv2.cvtColor(image_or_roi, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding for better results with varying lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)

    # Morphological Closing to connect small gaps in lines
    # Kernel size (e.g., 7x7) should be slightly larger than expected line breaks.
    kernel = np.ones((7, 7), np.uint8)
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Debug: Show the image after closing
    # cv2.imshow("Debug: Closed Thresh for Biggest Rect", closed_thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_rect_coords = None
    max_area = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) == 4 and cv2.contourArea(approx) > min_contour_area:
            x, y, w, h = cv2.boundingRect(approx)
            current_area = w * h

            if w > 0 and h > 0:
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 5.0:  # Filter out extremely wide or tall thin rectangles
                    if current_area > max_area:
                        max_area = current_area
                        biggest_rect_coords = (x, y, w, h)

    if biggest_rect_coords:
        x, y, w, h = biggest_rect_coords
        cropped_rectangle = image_or_roi[y: y + h, x: x + w]
        print(f"Found rectangle at: x={x}, y={y}, width={w}, height={h} (Area: {max_area})")
        return cropped_rectangle, biggest_rect_coords
    else:
        print(f"No significant rectangle found with min_area={min_contour_area}.")
        return None, None


def get_answer_grid_excluding_header(table_image):
    """
    Extracts the answer grid from a table image by identifying horizontal lines
    and cropping below the first distinct header line.

    Args:
        table_image (np.array): The cropped image of the entire table.

    Returns:
        np.array: The cropped image containing only the answer grid, or None.
    """
    if table_image is None:
        print("Input table image is None for header exclusion.")
        return None

    gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image where lines are clear
    binary_for_lines = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 10)

    # Debug: Show the binary image for line detection
    # cv2.imshow("Debug: Binary for Line Detection (Header Excl)", binary_for_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Detect horizontal lines using a morphological kernel
    # Kernel length should be significant to capture table lines.
    horizontal_kernel_len = max(20, table_image.shape[1] // 10)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))

    detected_horizontal_lines = cv2.morphologyEx(binary_for_lines, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Debug: Show detected horizontal lines
    # cv2.imshow("Debug: Detected Horizontal Lines (Header Excl)", detected_horizontal_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Use HoughLinesP to find precise Y-coordinates of horizontal lines
    # minLineLength: Minimum length of a line segment.
    # maxLineGap: Maximum allowed gap between points to link them.
    lines = cv2.HoughLinesP(detected_horizontal_lines, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)

    horizontal_lines_y = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # Ensure it's mostly horizontal
                horizontal_lines_y.append(y1)

    # Clean and sort line coordinates (remove very close duplicates)
    def clean_and_sort_lines(coords, tolerance=10):  # Tolerance for grouping close lines
        if not coords: return []
        coords.sort()
        cleaned = [coords[0]]
        for c in coords[1:]:
            if c - cleaned[-1] > tolerance:
                cleaned.append(c)
        return cleaned

    cleaned_horizontal_lines_y = clean_and_sort_lines(horizontal_lines_y,
                                                      tolerance=15)  # Increased tolerance for distinct rows

    # Debug: Draw detected lines on a copy of the table for visual verification
    # debug_lines_image = table_image.copy()
    # for y_coord in cleaned_horizontal_lines_y:
    #     cv2.line(debug_lines_image, (0, y_coord), (table_image.shape[1], y_coord), (255, 0, 0), 1)
    # cv2.imshow("Debug: Cleaned Horizontal Lines (Header Excl)", debug_lines_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if len(cleaned_horizontal_lines_y) < 2:
        print("Not enough distinct horizontal lines detected to separate header from grid.")
        return None

    # The top-most line is usually the very top of the table.
    # The second line is typically the bottom edge of the header row.
    # So, we want to crop starting from the y-coordinate of the second horizontal line.
    crop_top_y = cleaned_horizontal_lines_y[1]  # This cuts off the header row

    # The bottom of the grid is the last horizontal line found.
    crop_bottom_y = cleaned_horizontal_lines_y[-1]

    # Use the full width of the table_image for x-coordinates
    crop_left_x = 0
    crop_right_x = table_image.shape[1]

    # Ensure crop coordinates are within image bounds
    crop_top_y = max(0, crop_top_y)
    crop_bottom_y = min(table_image.shape[0], crop_bottom_y)

    if crop_bottom_y <= crop_top_y:  # Check for valid crop
        print("Invalid crop dimensions calculated.")
        return None

    cropped_answer_grid = table_image[crop_top_y: crop_bottom_y, crop_left_x: crop_right_x]

    return cropped_answer_grid


# --- Main execution ---
if __name__ == "__main__":
    image_file = 'omr2.png'

    # --- Step 1: Find the biggest rectangle (the entire main table) ---
    print("--- Step 1: Finding the Main Table ---")
    original_image = cv2.imread(image_file)
    if original_image is None:
        print(f"Error: Could not load image from {image_file}")
    else:
        # Use a large min_contour_area to get the overall document/table
        main_table_image, _ = find_biggest_rectangle_robust(original_image, min_contour_area=5000)

        if main_table_image is not None:
            cv2.imshow("Step 1: Main Table Extracted", main_table_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # --- Step 2: Extract the answer grid, excluding the header row ---
            print("\n--- Step 2: Extracting Answer Grid (excluding header) ---")
            answer_grid_only = get_answer_grid_excluding_header(main_table_image)

            if answer_grid_only is not None:
                cv2.imshow("Step 2: Answer Grid Only", answer_grid_only)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # Optional: Save the final result
                # cv2.imwrite("final_answer_grid_no_header.png", answer_grid_only)
                # print("Final answer grid saved as 'final_answer_grid_no_header.png'")
            else:
                print("Failed to extract the answer grid by excluding the header.")
        else:
            print("Main table not found, cannot proceed to extract answer grid.")