import cv2
import numpy as np

ANGLES_NUMBER = 4
EPSILON_CONSTANT = 0.02
CLOSE_KERNEL_SIZE = 10.0
CANNY_THRESHOLD_LOW = 75.0
CANNY_THRESHOLD_HIGH = 200.0
CUTOFF_THRESHOLD = 155.0
TRUNCATE_THRESHOLD = 150.0
NORMALIZATION_MIN_VALUE = 0.0
NORMALIZATION_MAX_VALUE = 255.0
BLURRING_KERNEL_SIZE = 5.0
DOWNSCALE_IMAGE_SIZE = 600.0
FIRST_MAX_CONTOURS = 10

def get_scanned_bitmap(image, x1, y1, x2, y2, x3, y3, x4, y4):
    rectangle = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32)
    dst_mat = perspective_transform(image, rectangle)
    return dst_mat

def get_contour_edge_points(temp_image):
    point2f = get_point(temp_image)
    if point2f is None:
        point2f = np.zeros((0, 0), dtype=np.float32)
    points = point2f.reshape(-1, 2)
    return points.tolist()

def get_point(image):
    src = image.copy()

    ratio = DOWNSCALE_IMAGE_SIZE / max(src.shape[1], src.shape[0])
    downscaled_size = (int(src.shape[1] * ratio), int(src.shape[0] * ratio))
    downscaled = cv2.resize(src, downscaled_size)
    largest_rectangle = detect_largest_quadrilateral(downscaled)

    if largest_rectangle is not None:
        return largest_rectangle['contour'].reshape(-1, 2) / ratio
    return None

def detect_largest_quadrilateral(src):
    
    destination = src.copy()

    cv2.imshow('Original Image', src)
    cv2.waitKey(0)

    # 
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY, 4)
    cv2.imshow('Gray Image', src)
    cv2.waitKey(0)

    # step 1
    src = cv2.GaussianBlur(src, (int(BLURRING_KERNEL_SIZE), int(BLURRING_KERNEL_SIZE)),0)
    src = cv2.normalize(src, None, NORMALIZATION_MIN_VALUE, NORMALIZATION_MAX_VALUE, cv2.NORM_MINMAX)

    cv2.imshow('Blurred Image', src)
    cv2.waitKey(0)

    # step 2
    _, src = cv2.threshold(src, TRUNCATE_THRESHOLD, NORMALIZATION_MAX_VALUE, cv2.THRESH_TRUNC)
    src = cv2.normalize(src, None, NORMALIZATION_MIN_VALUE, NORMALIZATION_MAX_VALUE, cv2.NORM_MINMAX)

    cv2.imshow('Truncated Image', src)
    cv2.waitKey(0)


    # step 3
    destination = cv2.Canny(src, int(CANNY_THRESHOLD_LOW), int(CANNY_THRESHOLD_HIGH))

    cv2.imshow('Canny Edge Detection', destination)
    cv2.waitKey(0)

    # step 4
    _, destination = cv2.threshold(destination, CUTOFF_THRESHOLD, NORMALIZATION_MAX_VALUE, cv2.THRESH_TOZERO)

    cv2.imshow('Cutoff Weak Edges', destination)
    cv2.waitKey(0)

    # step 5
    kernel = np.ones((int(CLOSE_KERNEL_SIZE), int(CLOSE_KERNEL_SIZE)), np.uint8)
    destination = cv2.morphologyEx(destination, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Morphological Closing', destination)
    cv2.waitKey(0)

    # Get only the 10 largest contours (each approximated to their convex hulls)
    largest_contour = find_largest_contours(destination)
    if largest_contour is not None:
        return find_quadrilateral(largest_contour)
    return None

def find_quadrilateral(contour_list):
    for c in contour_list:
        c2f = cv2.approxPolyDP(c, EPSILON_CONSTANT * cv2.arcLength(c, True), True)
        points = c2f.reshape(-1, 2)

        if c2f.shape[0] == ANGLES_NUMBER:
            found_points = sort_points(points)
            return {'contour': c2f, 'points': found_points}
        elif c2f.shape[0] == 5:
            shortest_distance = float('inf')
            shortest_point1 = None
            shortest_point2 = None

            diagonal = 0.0
            diagonal_point1 = None
            diagonal_point2 = None

            for i in range(4):
                for j in range(i + 1, 5):
                    d = distance(points[i], points[j])
                    if d < shortest_distance:
                        shortest_distance = d
                        shortest_point1 = points[i]
                        shortest_point2 = points[j]
                    if d > diagonal:
                        diagonal = d
                        diagonal_point1 = points[i]
                        diagonal_point2 = points[j]

            triangle_point_with_hypotenuse = list(
                filter(lambda p: p not in [shortest_point1, shortest_point2, diagonal_point1, diagonal_point2], points))[0]

            new_point = None
            if (triangle_point_with_hypotenuse[0] > shortest_point1[0] and
                    triangle_point_with_hypotenuse[0] > shortest_point2[0] and
                    triangle_point_with_hypotenuse[1] > shortest_point1[1] and
                    triangle_point_with_hypotenuse[1] > shortest_point2[1]):
                new_point = (min(shortest_point1[0], shortest_point2[0]), min(shortest_point1[1], shortest_point2[1]))
            elif (triangle_point_with_hypotenuse[0] < shortest_point1[0] and
                    triangle_point_with_hypotenuse[0] < shortest_point2[0] and
                    triangle_point_with_hypotenuse[1] > shortest_point1[1] and
                    triangle_point_with_hypotenuse[1] > shortest_point2[1]):
                new_point = (max(shortest_point1[0], shortest_point2[0]), min(shortest_point1[1], shortest_point2[1]))
            elif (triangle_point_with_hypotenuse[0] < shortest_point1[0] and
                    triangle_point_with_hypotenuse[0] < shortest_point2[0] and
                    triangle_point_with_hypotenuse[1] < shortest_point1[1] and
                    triangle_point_with_hypotenuse[1] < shortest_point2[1]):
                new_point = (max(shortest_point1[0], shortest_point2[0]), max(shortest_point1[1], shortest_point2[1]))
            elif (triangle_point_with_hypotenuse[0] > shortest_point1[0] and
                    triangle_point_with_hypotenuse[0] > shortest_point2[0] and
                    triangle_point_with_hypotenuse[1] < shortest_point1[1] and
                    triangle_point_with_hypotenuse[1] < shortest_point2[1]):
                new_point = (min(shortest_point1[0], shortest_point2[0]), max(shortest_point1[1], shortest_point2[1]))

            if new_point is not None:
                sorted_points = sort_points([triangle_point_with_hypotenuse, diagonal_point1,
                                                                diagonal_point2, new_point])
                new_approx = np.array(sorted_points, dtype=np.float32)
                return {'contour': new_approx, 'points': sorted_points}
    return None

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def sort_points(src):
    src_points = src.copy()
    result = [None, None, None, None]
    sum_comparator = lambda p: p[1] + p[0]
    diff_comparator = lambda p: p[1] - p[0]

    # top-left corner = minimal sum
    result[0] = min(src_points, key=sum_comparator)
    # bottom-right corner = maximal sum
    result[2] = max(src_points, key=sum_comparator)
    # top-right corner = minimal difference
    result[1] = min(src_points, key=diff_comparator)
    # bottom-left corner = maximal difference
    result[3] = max(src_points, key=diff_comparator)

    return np.array(result, dtype=np.float32)

def find_largest_contours(input_image):
    _, contours, _ = cv2.findContours(input_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the contours to their Convex Hulls i.e. removes minor nuances in the contour
    hull_list = [hull2points(cv2.convexHull(contour), contour) for contour in contours]

    if hull_list:
        hull_list.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        return hull_list[:min(len(hull_list), FIRST_MAX_CONTOURS)]
    return None

def hull2points(hull, contour):
    indexes = hull.flatten()
    points = contour[indexes]
    return points

def perspective_transform(image, rectangle):
    src_rect = np.array(rectangle, dtype=np.float32)
    dst_rect = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]],
                        dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(src_rect, dst_rect)
    result = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))
    return result
    
# Example usage:
img = cv2.imread('IMG_20231213_152323.jpg') #white bg
#img = cv2.imread('IMG_20231213_152313.jpg') 

points = get_contour_edge_points(img)

x1 = points.getValue(0).x
x2 = points.getValue(1).x
x3 = points.getValue(2).x
x4 = points.getValue(3).x
y1 = points.getValue(0).y
y2 = points.getValue(1).y
y3 = points.getValue(2).y
y4 = points.getValue(3).y

scanned_image = get_scanned_bitmap(img, x1, y1, x2, y2, x3, y3, x4, y4)

# Display the original and cropped images
cv2.imshow('Cropped Image', scanned_image)
cv2.waitKey(0)