from PIL import Image

def detect_selected_option(image_path: str) -> int | None:
    """
    Detects which option (1, 2, 3, or 4) is selected in an image.

    This function assumes the image contains four vertically stacked circular options,
    where the selected option has a significantly darker background compared to
    the unselected ones. It works by converting the image to grayscale and
    calculating the average pixel intensity within predefined regions of interest (ROIs)
    for each option. The option with the lowest average pixel value (darkest)
    is identified as selected.

    Args:
        image_path (str): The file path to the image (e.g., 'question_23.png').

    Returns:
        int or None: The number of the selected option (1-4), or None if no
                     distinctly selected option is found based on the defined
                     thresholds.

    Instructions for Use:
    1.  **Install Pillow:** If you don't have Pillow installed, run:
        `pip install Pillow`
    2.  **Place your image:** Make sure the image file (e.g., 'question_23.png')
        is in the same directory as your Python script, or provide its full path.
    3.  **Adjust ROIs (if necessary):** The `rois` coordinates are estimated
        based on the example image. If your images have different resolutions,
        layouts, or spacing between options, you might need to adjust the
        `x_left`, `x_right`, `y_upper`, `y_lower` calculations or directly
        define the `rois` list.
        -   `x_left`, `x_right`: Define the horizontal bounds of the options.
            They are set to take a central slice of the image width.
        -   `option_height`: Divides the total image height by 4.
        -   `vertical_padding`: Used to shrink the ROI slightly within each
            quarter to avoid capturing parts of surrounding elements or borders.
    4.  **Adjust Thresholds (if necessary):** The `threshold_difference_ratio`
        and the fallback `min_max_diff_threshold` are used to determine if
        the darkest option is *distinct enough* to be considered selected.
        If the script incorrectly identifies options, or fails to identify
        a clearly selected option, these values might need fine-tuning.
        -   A lower average pixel value means a darker region.
        -   `threshold_difference_ratio`: Checks if the darkest option is
            significantly darker than the *second* darkest option.
        -   `min_max_diff_threshold`: A fallback check to see if there's
            a large overall difference between the darkest and lightest option.
    """
    try:
        # Open the image and convert it to grayscale for easier intensity comparison.
        # Grayscale pixels range from 0 (black) to 255 (white).
        img = Image.open(image_path).convert("L")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred while opening or processing the image: {e}")
        return None

    width, height = img.size
    print(f"Image loaded: {image_path}, Dimensions: {width}x{height}")

    # --- Define Regions of Interest (ROIs) for each option ---
    # These coordinates define the rectangular areas where each option is expected.
    # They are dynamically calculated based on image dimensions for some adaptability.

    # Horizontal bounds: Take a central slice of the image's width.
    # Adjust `roi_width_factor` to make the slice wider or narrower.
    roi_width_factor = 0.4 # Takes 40% of the image width around the center
    x_center = width // 2
    roi_width = int(width * roi_width_factor)
    x_left = max(0, x_center - roi_width // 2)
    x_right = min(width, x_center + roi_width // 2)

    # Vertical bounds: Divide the image height into 4 sections for 4 options.
    # `vertical_padding_factor` ensures the ROI is slightly smaller than the full quarter,
    # avoiding unwanted edges or text outside the option circle.
    option_section_height = height // 4
    vertical_padding_factor = 0.15 # 15% padding from top/bottom of each option's section

    rois = [] # List to store (left, upper, right, lower) tuples for each ROI
    for i in range(4): # For options 1 through 4
        y_upper = i * option_section_height + int(option_section_height * vertical_padding_factor)
        y_lower = (i + 1) * option_section_height - int(option_section_height * vertical_padding_factor)
        # Ensure coordinates are within image bounds
        y_upper = max(0, y_upper)
        y_lower = min(height, y_lower)
        rois.append((x_left, y_upper, x_right, y_lower))

    avg_pixel_values = []
    for i, roi in enumerate(rois):
        if roi[2] <= roi[0] or roi[3] <= roi[1]: # Check for valid ROI dimensions
            print(f"Warning: ROI for option {i+1} is invalid: {roi}. Skipping.")
            avg_pixel_values.append(float('inf')) # Mark as very high so it's not selected
            continue

        cropped_img = img.crop(roi)
        # Calculate the average pixel value for the current ROI.
        # Lower value = darker region.
        pixel_data = list(cropped_img.getdata())
        if not pixel_data: # Handle case where cropped image might be empty
            avg_pixel = float('inf')
        else:
            avg_pixel = sum(pixel_data) / len(pixel_data)
        avg_pixel_values.append(avg_pixel)
        print(f"Option {i+1} ROI: {roi}, Average pixel value: {avg_pixel:.2f}")

    if not avg_pixel_values or all(val == float('inf') for val in avg_pixel_values):
        print("No valid ROIs found or all ROIs are empty/invalid.")
        return None

    # Find the option with the minimum average pixel value (darkest region).
    min_avg_value = float('inf')
    selected_option_index = -1

    for i, avg_val in enumerate(avg_pixel_values):
        if avg_val < min_avg_value:
            min_avg_value = avg_val
            selected_option_index = i

    # If no option was found (e.g., all were infinity), return None
    if selected_option_index == -1:
        print("Could not determine a selected option.")
        return None

    # --- Heuristic for determining distinct selection ---
    # This helps ensure that the darkest region is *significantly* darker,
    # not just slightly darker due to noise or image variations.

    sorted_values = sorted(val for val in avg_pixel_values if val != float('inf'))

    if not sorted_values: # No valid pixel values
        return None

    # If there's only one valid option, and it's the darkest, assume it's selected.
    if len(sorted_values) == 1:
        return selected_option_index + 1

    # Threshold 1: The darkest option's average value should be significantly
    # lower than the second darkest option.
    # This ratio is the percentage difference.
    threshold_difference_ratio = 0.35 # Selected option must be 35% darker than the next one
    # For example, if sorted_values[0] = 50 (dark) and sorted_values[1] = 100 (lighter)
    # then (100 - 50) / 100 = 0.5 (50% difference), which is > 0.35.
    if min_avg_value < sorted_values[1] and \
       (sorted_values[1] - min_avg_value) / sorted_values[1] > threshold_difference_ratio:
        print(f"Selected option {selected_option_index + 1} (Avg: {min_avg_value:.2f}) is significantly darker than the next darkest (Avg: {sorted_values[1]:.2f}).")
        return selected_option_index + 1

    # Threshold 2: Fallback - if the ratio isn't met, check if there's a large
    # absolute difference between the darkest and the brightest option.
    min_max_diff_threshold = 40 # An absolute difference of 40 in grayscale values (out of 255)
    max_avg_value = max(val for val in avg_pixel_values if val != float('inf'))

    if (max_avg_value - min_avg_value) > min_max_diff_threshold:
        print(f"Selected option {selected_option_index + 1} (Avg: {min_avg_value:.2f}) has a large absolute difference from the brightest option (Avg: {max_avg_value:.2f}).")
        return selected_option_index + 1

    print("No distinctly selected option found based on thresholds.")
    return None

# --- Example Usage ---
if __name__ == "__main__":
    # Replace 'question_23.png' with the actual path to your image file.
    image_file = 'answer_columns/question_23.png'
    selected = detect_selected_option(image_file)

    if selected is not None:
        print(f"\nThe selected option is: {selected}")
    else:
        print("\nCould not determine the selected option.")

    # You can test with a hypothetical image path that doesn't exist
    # print("\nTesting with a non-existent image:")
    # selected_none = detect_selected_option("non_existent_image.png")
    # if selected_none is None:
    #     print("Correctly handled non-existent image.")
