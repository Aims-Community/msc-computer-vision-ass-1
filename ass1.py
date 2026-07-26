from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def resolve_image_path(path):
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(__file__).resolve().parent / path_obj
    return str(path_obj.resolve())


def create_panorama(img1_path, img2_path):
    img1_path = resolve_image_path(img1_path)
    img2_path = resolve_image_path(img2_path)
    print(f"Creating panorama from images: {img1_path} and {img2_path}")
    # Load images in color for display, and grayscale for SIFT processing
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not read one or both images. Checked: {img1_path} and {img2_path}")
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 1 & 2. Initialize SIFT detector, detect features, and extract descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    print(f"Detected {len(kp1)} keypoints in Image 1 and {len(kp2)} in Image 2.")

    # 3. Perform matching using Brute-Force Matcher with k=2 for ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. Apply Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # 0.75 is the standard threshold
            good_matches.append(m)
            
    print(f"Good matches after Lowe's Ratio Test: {len(good_matches)}")

    # Optional: Visualize matching keypoints
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('matches.jpg', img_matches)

    # 5. Estimate Homography matrix using RANSAC
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # H transforms points from img1 to img2's coordinate space
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print("Estimated Homography Matrix:\n", H)
    else:
        raise AssertionError("Not enough matches found to compute homography.")

    # 6. Produce a panorama (Warping and Blending)
    # Determine dimensions for the canvas
    width = img1.shape[1] + img2.shape[1]
    height = max(img1.shape[0], img2.shape[0])
    
    # Warp image 1 using the homography matrix
    panorama = cv2.warpPerspective(img1, H, (width, height))
    
    # Place image 2 onto the canvas (assuming img2 is the base frame here)
    # Note: Depending on your input ordering, you may need to adjust the placement or invert H
    panorama[0:img2.shape[0], 0:img2.shape[1]] = img2

    # Crop trailing black pixels to clean up the canvas
    gray_pan = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_pan, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    panorama_cropped = panorama[y:y+h, x:x+w]

    cv2.imwrite('panorama_result.jpg', panorama_cropped)
    print("Panorama generated successfully and saved as 'panorama_result.jpg'")

# Execute code (Replace with your actual image file names)
create_panorama('testImages/left.png', 'testImages/right.png')