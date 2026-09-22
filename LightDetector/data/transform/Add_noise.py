import cv2
import numpy as np
import random

def Noise_Polygon(meta):
    image = meta["img"]
    # print(image.shape)
    # print(f'shape of image in noise add: {image.shape}')
    # Check if the image is loaded correctly
    if image is None:
        print("Error: Unable to load image. Please check the image path and file.")
        exit()  # Exit the program if image is not loaded
        
    bounding_boxes = []
    # print(f'meta keys: {meta.keys()}')
    bboxes = meta["gt_bboxes"].tolist()
    
    for box in bboxes:
        bounding_boxes.append(tuple(box))
    # print("bboxes", bounding_boxes)
    # Create a mask to track the areas where boundary boxes are placed
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
 
    # Fill the mask where the bounding boxes are
    for (x1, y1, x2, y2) in bounding_boxes:
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        mask[y1:y2, x1:x2] = 255  # Set the area of the bounding box to 255
    blur_ksize = (25, 25)  # Gaussian blur kernel size

    # Function to generate random polygon points
    def generate_polygon(x, y, width, height, sides):
        angle_step = 2 * np.pi / sides
        points = []
        for i in range(sides):
            angle = i * angle_step
            x_offset = int(np.cos(angle) * width / 2)
            y_offset = int(np.sin(angle) * height / 2)
            points.append((x + x_offset, y + y_offset))
        return np.array(points, dtype=np.int32)

    # Try to place multiple polygons where there are no bounding boxes
    polygons_placed = 0
    max_polygons = random.randint(1,10)  # Increased number of polygons
    max_attempts = 10  # Limit attempts to place polygons

    # Copy the original image to update with polygons
    image_with_polygons = image.copy()

    while polygons_placed < max_polygons and max_attempts > 0:
        # Randomly choose a position where there is no bounding box or previous polygon
        polygon_sides = random.randint(5, 8)  # Random number of sides for the polygon (e.g., 6 to 10)
        polygon_height = random.randint(30,100) # Increased height
        polygon_width = random.randint(40,300)   # Increased width
        # print(polygon_height, polygon_width)
        poly_width_range = image.shape[1] - polygon_width-40
        poly_height_range = image.shape[0] - polygon_height-40
        if poly_width_range <=0 or poly_height_range<=0:
            # print('neg')
            break
        x = random.randint(0, poly_width_range)
        y = random.randint(0, poly_height_range)
        polygon_points = generate_polygon(x, y, polygon_width, polygon_height, polygon_sides)
    
        # Create a temporary mask for the polygon
        polygon_mask = np.zeros_like(image)
        ch1 =  random.randint(200,255) 
        cv2.fillPoly(polygon_mask, [polygon_points], (ch1, ch1, ch1))

        # Check if the polygon overlaps with the bounding boxes or any previously placed polygons
        # If the polygon's mask overlaps with the current mask, it means the area is blocked
        if np.any(np.bitwise_and(mask, polygon_mask[:, :, 0])) == 0:  # No overlap
            # Apply the Gaussian blur to the region inside the polygon
            blurred_area = cv2.GaussianBlur(polygon_mask, blur_ksize, 0)

            # Add the blurred area to the image
            image_with_polygons = cv2.addWeighted(image_with_polygons, 1, blurred_area, 0.5, 0)

            # Update the mask to include the newly placed polygon (to avoid overlap with other polygons)
            cv2.fillPoly(mask, [polygon_points], 255)

            polygons_placed += 1

        max_attempts -= 1  # Prevent infinite loop in case no space is found
    
    meta["img"] = image_with_polygons
    return meta
