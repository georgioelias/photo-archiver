"""
Perspective correction module for laminated/polaroid photographs.
Implements border detection and perspective transformation.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PerspectiveResult:
    """Result from perspective correction processing."""
    image: np.ndarray
    corners_detected: Optional[np.ndarray]
    original_corners: Optional[np.ndarray]
    crop_region: Optional[Tuple[int, int, int, int]]
    method_used: str
    confidence: float
    processing_time: float


class PerspectiveCorrector:
    """
    Robust perspective correction for digitized photographs.
    
    Supports:
    - Contour-based border detection
    - Hough line detection (fallback)
    - Corner detection
    - Auto-crop for polaroid-style photos
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize perspective corrector with configuration.
        
        Args:
            config: Perspective correction configuration dictionary
        """
        self.config = config or {}
        self.default_config = {
            "border_detection": "auto",
            "method": "contour",
            "margin_percent": 2,
            "min_area_ratio": 0.3,
            "canny_low": 50,
            "canny_high": 150,
            "blur_kernel": 5,
            "epsilon_factor": 0.02,
            "detect_inner_photo": True,
        }
        # Merge with defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates: top-left has smallest, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Difference: top-right has smallest, bottom-left has largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _calculate_destination_size(self, corners: np.ndarray) -> Tuple[int, int]:
        """
        Calculate destination size maintaining aspect ratio.
        
        Args:
            corners: Ordered corner points
            
        Returns:
            Tuple of (width, height)
        """
        # Calculate widths
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = int(max(width_top, width_bottom))
        
        # Calculate heights
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        height = int(max(height_left, height_right))
        
        return width, height
    
    def detect_contour(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect photo border using contour detection.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (corners, confidence)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur_size = self.config["blur_kernel"]
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Edge detection with auto threshold
        median = np.median(blurred)
        low = int(max(0, (1.0 - 0.33) * median))
        high = int(min(255, (1.0 + 0.33) * median))
        edges = cv2.Canny(blurred, low, high)
        
        # Dilate edges to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0.0
        
        # Filter contours by area
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.config["min_area_ratio"]
        
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                valid_contours.append((cnt, area))
        
        if not valid_contours:
            return None, 0.0
        
        # Sort by area, take largest
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        largest_contour = valid_contours[0][0]
        
        # Approximate to polygon
        epsilon = self.config["epsilon_factor"] * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # We need exactly 4 points for perspective transform
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            confidence = valid_contours[0][1] / image_area
            return self._order_points(corners), confidence
        
        # If not 4 points, try to find 4 corners from convex hull
        hull = cv2.convexHull(largest_contour)
        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) >= 4:
            # Find 4 most extreme points
            corners = self._find_extreme_corners(approx.reshape(-1, 2))
            confidence = valid_contours[0][1] / image_area * 0.8  # Lower confidence
            return self._order_points(corners), confidence
        
        return None, 0.0
    
    def _find_extreme_corners(self, points: np.ndarray) -> np.ndarray:
        """
        Find 4 extreme corner points from a set of points.
        
        Args:
            points: Array of points
            
        Returns:
            Array of 4 corner points
        """
        # Find extreme points
        sum_pts = points.sum(axis=1)
        diff_pts = np.diff(points, axis=1).flatten()
        
        top_left_idx = np.argmin(sum_pts)
        bottom_right_idx = np.argmax(sum_pts)
        top_right_idx = np.argmin(diff_pts)
        bottom_left_idx = np.argmax(diff_pts)
        
        corners = np.array([
            points[top_left_idx],
            points[top_right_idx],
            points[bottom_right_idx],
            points[bottom_left_idx]
        ], dtype=np.float32)
        
        return corners
    
    def detect_hough_lines(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect photo border using Hough line detection (fallback method).
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (corners, confidence)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=min(image.shape[:2])//4,
                                maxLineGap=10)
        
        if lines is None or len(lines) < 4:
            return None, 0.0
        
        # Classify lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 30 or angle > 150:
                horizontal_lines.append(line[0])
            elif 60 < angle < 120:
                vertical_lines.append(line[0])
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None, 0.0
        
        # Find top/bottom horizontal lines and left/right vertical lines
        h_lines = np.array(horizontal_lines)
        v_lines = np.array(vertical_lines)
        
        # Sort horizontal by y-coordinate
        h_y_coords = (h_lines[:, 1] + h_lines[:, 3]) / 2
        h_sorted_idx = np.argsort(h_y_coords)
        top_line = h_lines[h_sorted_idx[0]]
        bottom_line = h_lines[h_sorted_idx[-1]]
        
        # Sort vertical by x-coordinate
        v_x_coords = (v_lines[:, 0] + v_lines[:, 2]) / 2
        v_sorted_idx = np.argsort(v_x_coords)
        left_line = v_lines[v_sorted_idx[0]]
        right_line = v_lines[v_sorted_idx[-1]]
        
        # Find intersections
        corners = np.array([
            self._line_intersection(top_line, left_line),
            self._line_intersection(top_line, right_line),
            self._line_intersection(bottom_line, right_line),
            self._line_intersection(bottom_line, left_line)
        ], dtype=np.float32)
        
        # Validate corners are within image bounds
        h, w = image.shape[:2]
        margin = max(h, w) * 0.1
        
        for corner in corners:
            if (corner[0] < -margin or corner[0] > w + margin or
                corner[1] < -margin or corner[1] > h + margin):
                return None, 0.0
        
        # Clip to image bounds
        corners[:, 0] = np.clip(corners[:, 0], 0, w-1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h-1)
        
        confidence = 0.6  # Lower confidence for Hough method
        return self._order_points(corners), confidence
    
    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
        """
        Calculate intersection point of two lines.
        
        Args:
            line1: First line as (x1, y1, x2, y2)
            line2: Second line as (x1, y1, x2, y2)
            
        Returns:
            Intersection point (x, y)
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            # Lines are parallel, return midpoint
            return np.array([(x1 + x3) / 2, (y1 + y3) / 2])
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return np.array([x, y])
    
    def apply_perspective_transform(self, image: np.ndarray, 
                                     corners: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to straighten the image.
        
        Args:
            image: BGR input image
            corners: Ordered corner points
            
        Returns:
            Perspective-corrected image
        """
        # Calculate destination size
        width, height = self._calculate_destination_size(corners)
        
        # Ensure minimum dimensions
        width = max(width, 100)
        height = max(height, 100)
        
        # Define destination points
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Get transformation matrix
        matrix = cv2.getPerspectiveTransform(corners, dst)
        
        # Apply transformation
        corrected = cv2.warpPerspective(image, matrix, (width, height))
        
        return corrected
    
    def auto_detect_photo_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the actual photo content area (for polaroid-style photos).
        
        Args:
            image: BGR input image
            
        Returns:
            Crop coordinates (x, y, w, h) or None if not applicable
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check if there's a bright border
        border_size = min(h, w) // 15
        
        # Sample border regions
        top_mean = np.mean(gray[:border_size, :])
        bottom_mean = np.mean(gray[-border_size:, :])
        left_mean = np.mean(gray[:, :border_size])
        right_mean = np.mean(gray[:, -border_size:])
        
        border_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4
        center_mean = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
        
        # If border is significantly brighter, detect inner photo
        if border_mean - center_mean > 30:
            # Find inner photo region using edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find contour closest to center with significant area
                center = np.array([w // 2, h // 2])
                best_contour = None
                best_score = float('inf')
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < h * w * 0.2:  # Too small
                        continue
                    
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        dist = np.linalg.norm(np.array([cx, cy]) - center)
                        score = dist - area * 0.001  # Prefer larger, centered contours
                        
                        if score < best_score:
                            best_score = score
                            best_contour = cnt
                
                if best_contour is not None:
                    x, y, w_crop, h_crop = cv2.boundingRect(best_contour)
                    
                    # Add small margin
                    margin = 5
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w_crop = min(w - x, w_crop + 2 * margin)
                    h_crop = min(h - y, h_crop + 2 * margin)
                    
                    return (x, y, w_crop, h_crop)
        
        return None
    
    def apply_margin_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Apply margin crop to remove edge artifacts.
        
        Args:
            image: BGR input image
            
        Returns:
            Cropped image
        """
        margin_percent = self.config["margin_percent"]
        if margin_percent <= 0:
            return image
        
        h, w = image.shape[:2]
        margin_x = int(w * margin_percent / 100)
        margin_y = int(h * margin_percent / 100)
        
        # Ensure we don't crop too much
        margin_x = min(margin_x, w // 10)
        margin_y = min(margin_y, h // 10)
        
        cropped = image[margin_y:h-margin_y, margin_x:w-margin_x]
        
        return cropped
    
    def process(self, image: np.ndarray) -> PerspectiveResult:
        """
        Main processing function for perspective correction.
        
        Args:
            image: BGR input image
            
        Returns:
            PerspectiveResult with corrected image and metadata
        """
        import time
        start_time = time.time()
        
        original_corners = None
        method_used = "none"
        confidence = 0.0
        crop_region = None
        
        # Try contour detection first
        corners, conf = self.detect_contour(image)
        
        if corners is not None and conf > 0.3:
            original_corners = corners.copy()
            method_used = "contour"
            confidence = conf
        else:
            # Fallback to Hough lines
            corners, conf = self.detect_hough_lines(image)
            if corners is not None:
                original_corners = corners.copy()
                method_used = "hough"
                confidence = conf
        
        # Apply perspective transform if corners detected
        if corners is not None and confidence > 0.2:
            result_image = self.apply_perspective_transform(image, corners)
        else:
            result_image = image.copy()
            method_used = "none"
        
        # Detect and crop inner photo region if configured
        if self.config["detect_inner_photo"]:
            crop_coords = self.auto_detect_photo_region(result_image)
            if crop_coords:
                x, y, w, h = crop_coords
                result_image = result_image[y:y+h, x:x+w]
                crop_region = crop_coords
        
        # Apply margin crop
        result_image = self.apply_margin_crop(result_image)
        
        return PerspectiveResult(
            image=result_image,
            corners_detected=corners,
            original_corners=original_corners,
            crop_region=crop_region,
            method_used=method_used,
            confidence=confidence,
            processing_time=time.time() - start_time
        )


def correct_perspective(image: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
    """
    Convenience function to correct perspective.
    
    Args:
        image: BGR input image
        config: Optional configuration
        
    Returns:
        Perspective-corrected image
    """
    corrector = PerspectiveCorrector(config)
    result = corrector.process(image)
    return result.image


@dataclass
class PolaroidCropResult:
    """Result from polaroid content cropping."""
    image: np.ndarray
    crop_region: Optional[Tuple[int, int, int, int]]
    border_detected: bool
    content_ratio: float
    detection_method: str
    processing_time: float


class PolaroidCropper:
    """
    Robust photo content extraction from polaroid/laminated frames.
    
    PRIMARY STRATEGY: Strict white border detection for polaroids.
    Polaroid frames have distinctive white borders that are:
    - Very bright (>200 luminance typically)
    - Low saturation (near white/off-white)
    - Uniform color with low variance
    
    Fallback strategies:
    1. Color variance analysis
    2. Edge density analysis
    3. Gradient-based border detection
    4. Contour-based detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize polaroid cropper.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.default_config = {
            "min_content_ratio": 0.25,  # Min ratio of content to total area
            "white_threshold": 200,     # Strict threshold for white detection (raised for polaroids)
            "saturation_threshold": 40, # Max saturation for white areas
            "margin_pixels": 5,         # Extra margin to crop
            "variance_threshold": 500,  # Color variance threshold for content detection
            "edge_density_ratio": 2.0,  # Content should have 2x more edges than border
            "min_border_width": 0.03,   # Minimum border width as ratio of image dimension
            "strict_white_detection": True,  # Enable strict white border detection
        }
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def _analyze_region_properties(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image to determine if it has a frame structure.
        
        Returns dict with border analysis metrics.
        """
        h, w = gray.shape
        border_size = min(h, w) // 12  # Slightly larger sample
        
        # Sample border regions
        regions = {
            'top': gray[:border_size, border_size:-border_size],
            'bottom': gray[-border_size:, border_size:-border_size],
            'left': gray[border_size:-border_size, :border_size],
            'right': gray[border_size:-border_size, -border_size:],
            'center': gray[h//4:3*h//4, w//4:3*w//4]
        }
        
        # Calculate variance and mean for each region
        stats = {}
        for name, region in regions.items():
            if region.size > 0:
                stats[name] = {
                    'mean': float(np.mean(region)),
                    'std': float(np.std(region)),
                    'variance': float(np.var(region))
                }
            else:
                stats[name] = {'mean': 128, 'std': 50, 'variance': 2500}
        
        # Border uniformity check - borders should be more uniform than content
        border_variances = [stats[r]['variance'] for r in ['top', 'bottom', 'left', 'right']]
        center_variance = stats['center']['variance']
        
        avg_border_variance = np.mean(border_variances)
        
        # Color analysis (for colored borders)
        if len(image.shape) == 3:
            border_samples = [
                image[:border_size, border_size:-border_size],
                image[-border_size:, border_size:-border_size],
                image[border_size:-border_size, :border_size],
                image[border_size:-border_size, -border_size:]
            ]
            center_sample = image[h//4:3*h//4, w//4:3*w//4]
            
            # Calculate color variance
            border_color_var = np.mean([np.var(s) for s in border_samples if s.size > 0])
            center_color_var = np.var(center_sample) if center_sample.size > 0 else 0
        else:
            border_color_var = avg_border_variance
            center_color_var = center_variance
        
        return {
            'stats': stats,
            'avg_border_variance': avg_border_variance,
            'center_variance': center_variance,
            'variance_ratio': center_variance / max(avg_border_variance, 1),
            'border_color_var': border_color_var,
            'center_color_var': center_color_var,
            'has_frame_structure': center_variance > avg_border_variance * 1.5
        }
    
    def _detect_strict_white_borders(self, image: np.ndarray, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        STRICT white border detection specifically designed for polaroid frames.
        
        Polaroids have asymmetric borders - the signature strip (thick border) is 
        typically 2-3x wider than the other three borders. This method:
        1. Detects all white borders
        2. Identifies the thick border (signature strip)
        3. Crops to the INNER content, excluding all white borders completely
        """
        h, w = image.shape[:2]
        
        # Convert to HSV for better white detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # White detection: high brightness AND low saturation
        # Use adaptive threshold based on border samples
        border_sample_size = min(h, w) // 8
        
        # Sample corners to estimate white threshold
        corners = [
            value[:border_sample_size, :border_sample_size],
            value[:border_sample_size, -border_sample_size:],
            value[-border_sample_size:, :border_sample_size],
            value[-border_sample_size:, -border_sample_size:]
        ]
        corner_brightness = np.mean([np.mean(c) for c in corners])
        
        # Adaptive threshold - if corners are bright, they're likely white borders
        white_threshold = max(160, min(200, corner_brightness - 20))
        
        # Create white mask
        white_mask = (value > white_threshold) & (saturation < 70)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Scan from each edge to find where content begins
        # Use stricter criteria - look for significant non-white content
        
        # LEFT boundary
        left_boundary = 0
        for col in range(min(w // 2, w)):
            col_white_ratio = np.mean(white_mask[:, col])
            col_brightness = np.mean(value[:, col])
            col_saturation = np.mean(saturation[:, col])
            # Content has lower brightness OR higher saturation (color)
            if col_white_ratio < 0.4 and (col_brightness < white_threshold - 30 or col_saturation > 40):
                left_boundary = col
                break
        
        # RIGHT boundary
        right_boundary = w
        for col in range(w - 1, max(w // 2, 0), -1):
            col_white_ratio = np.mean(white_mask[:, col])
            col_brightness = np.mean(value[:, col])
            col_saturation = np.mean(saturation[:, col])
            if col_white_ratio < 0.4 and (col_brightness < white_threshold - 30 or col_saturation > 40):
                right_boundary = col
                break
        
        # TOP boundary
        top_boundary = 0
        for row in range(min(h // 2, h)):
            row_white_ratio = np.mean(white_mask[row, :])
            row_brightness = np.mean(value[row, :])
            row_saturation = np.mean(saturation[row, :])
            if row_white_ratio < 0.4 and (row_brightness < white_threshold - 30 or row_saturation > 40):
                top_boundary = row
                break
        
        # BOTTOM boundary - scan more aggressively for the thick border
        bottom_boundary = h
        for row in range(h - 1, max(h // 2, 0), -1):
            row_white_ratio = np.mean(white_mask[row, :])
            row_brightness = np.mean(value[row, :])
            row_saturation = np.mean(saturation[row, :])
            if row_white_ratio < 0.4 and (row_brightness < white_threshold - 30 or row_saturation > 40):
                bottom_boundary = row
                break
        
        # Calculate border widths
        border_left = left_boundary
        border_right = w - right_boundary
        border_top = top_boundary
        border_bottom = h - bottom_boundary
        
        # Identify the thick border (signature strip) - it should be significantly larger
        borders = {'left': border_left, 'right': border_right, 'top': border_top, 'bottom': border_bottom}
        max_border = max(borders.values())
        min_meaningful_border = min(h, w) * 0.02
        
        # Check we have at least some borders
        meaningful_borders = sum(1 for b in borders.values() if b > min_meaningful_border)
        if meaningful_borders < 2:
            return None
        
        # Content dimensions
        content_width = right_boundary - left_boundary
        content_height = bottom_boundary - top_boundary
        
        if content_width < w * 0.2 or content_height < h * 0.2:
            return None
        
        if content_width > w * 0.98 or content_height > h * 0.98:
            return None
        
        # Add inward margin to ensure clean crop (no white remnants)
        # Use larger margin for the thick border side
        base_margin = max(5, int(min(h, w) * 0.008))  # 0.8% minimum
        
        # Apply margins - slightly larger to ensure no white edges
        x = left_boundary + base_margin
        y = top_boundary + base_margin
        cw = content_width - 2 * base_margin
        ch = content_height - 2 * base_margin
        
        # Bounds check
        x = max(0, min(w - 20, x))
        y = max(0, min(h - 20, y))
        cw = max(20, min(w - x, cw))
        ch = max(20, min(h - y, ch))
        
        return (x, y, cw, ch)
    
    def _detect_white_border_with_gradient(self, image: np.ndarray, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect white borders using gradient analysis at border transitions.
        
        Uses cumulative brightness tracking to find where the white border
        transitions to photo content. Works well for thick borders.
        """
        h, w = image.shape[:2]
        
        # Use HSV for saturation check as well
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Estimate white threshold from edges
        edge_brightness = np.mean([
            np.mean(value[:, :10]),
            np.mean(value[:, -10:]),
            np.mean(value[:10, :]),
            np.mean(value[-10:, :])
        ])
        white_threshold = max(150, edge_brightness - 30)
        
        # LEFT boundary - find where bright white ends
        left_boundary = 0
        for col in range(min(w // 2, w)):
            col_brightness = np.mean(value[:, col])
            col_saturation = np.mean(saturation[:, col])
            
            # White ends when brightness drops significantly OR saturation increases
            if col_brightness < white_threshold - 20 or col_saturation > 50:
                left_boundary = col
                break
        
        # RIGHT boundary
        right_boundary = w
        for col in range(w - 1, max(w // 2, 0), -1):
            col_brightness = np.mean(value[:, col])
            col_saturation = np.mean(saturation[:, col])
            
            if col_brightness < white_threshold - 20 or col_saturation > 50:
                right_boundary = col
                break
        
        # TOP boundary
        top_boundary = 0
        for row in range(min(h // 2, h)):
            row_brightness = np.mean(value[row, :])
            row_saturation = np.mean(saturation[row, :])
            
            if row_brightness < white_threshold - 20 or row_saturation > 50:
                top_boundary = row
                break
        
        # BOTTOM boundary - likely the thick border, be thorough
        bottom_boundary = h
        for row in range(h - 1, max(h // 2, 0), -1):
            row_brightness = np.mean(value[row, :])
            row_saturation = np.mean(saturation[row, :])
            
            if row_brightness < white_threshold - 20 or row_saturation > 50:
                bottom_boundary = row
                break
        
        # Validate
        content_width = right_boundary - left_boundary
        content_height = bottom_boundary - top_boundary
        
        min_border = int(min(h, w) * 0.02)
        border_left = left_boundary
        border_right = w - right_boundary
        border_top = top_boundary
        border_bottom = h - bottom_boundary
        
        meaningful_borders = sum(1 for b in [border_left, border_right, border_top, border_bottom] if b > min_border)
        
        if meaningful_borders < 2:
            return None
        
        if content_width < w * 0.2 or content_height < h * 0.2:
            return None
        
        # Apply margin to ensure clean crop
        margin = max(5, int(min(h, w) * 0.008))
        
        x = max(0, left_boundary + margin)
        y = max(0, top_boundary + margin)
        cw = max(20, content_width - 2 * margin)
        ch = max(20, content_height - 2 * margin)
        
        # Bounds check
        if x + cw > w:
            cw = w - x
        if y + ch > h:
            ch = h - y
        
        return (x, y, cw, ch)
    
    def _detect_by_gradient_transition(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect content by finding strong gradient transitions (border edges).
        """
        h, w = gray.shape
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Project gradients onto rows and columns
        row_gradient = np.mean(grad_mag, axis=1)
        col_gradient = np.mean(grad_mag, axis=0)
        
        # Find peaks in gradient (indicates border transition)
        threshold = np.mean(grad_mag) * 1.5
        
        # Find left boundary (first significant gradient peak from left)
        left = 0
        for i in range(w // 4):
            if col_gradient[i] > threshold:
                left = i
                break
        
        # Find right boundary (first significant gradient peak from right)
        right = w
        for i in range(w - 1, 3 * w // 4, -1):
            if col_gradient[i] > threshold:
                right = i
                break
        
        # Find top boundary
        top = 0
        for i in range(h // 4):
            if row_gradient[i] > threshold:
                top = i
                break
        
        # Find bottom boundary
        bottom = h
        for i in range(h - 1, 3 * h // 4, -1):
            if row_gradient[i] > threshold:
                bottom = i
                break
        
        # Validate
        content_h = bottom - top
        content_w = right - left
        
        if content_h > h * 0.3 and content_w > w * 0.3:
            return (left, top, content_w, content_h)
        
        return None
    
    def _detect_by_edge_density(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect content region by analyzing edge density differences.
        Content areas typically have more edges than uniform borders.
        """
        h, w = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Calculate cumulative edge density from borders inward
        # Horizontal scan
        col_density = np.sum(edges, axis=0) / h
        row_density = np.sum(edges, axis=1) / w
        
        # Smooth the density profiles
        kernel = np.ones(5) / 5
        col_density_smooth = np.convolve(col_density, kernel, mode='same')
        row_density_smooth = np.convolve(row_density, kernel, mode='same')
        
        # Find where edge density increases significantly (content starts)
        density_threshold = np.mean(edges) * self.config["edge_density_ratio"]
        
        # Find boundaries
        left = 0
        for i in range(w // 3):
            if col_density_smooth[i] > density_threshold:
                left = max(0, i - 5)
                break
        
        right = w
        for i in range(w - 1, 2 * w // 3, -1):
            if col_density_smooth[i] > density_threshold:
                right = min(w, i + 5)
                break
        
        top = 0
        for i in range(h // 3):
            if row_density_smooth[i] > density_threshold:
                top = max(0, i - 5)
                break
        
        bottom = h
        for i in range(h - 1, 2 * h // 3, -1):
            if row_density_smooth[i] > density_threshold:
                bottom = min(h, i + 5)
                break
        
        content_h = bottom - top
        content_w = right - left
        
        if content_h > h * 0.3 and content_w > w * 0.3:
            return (left, top, content_w, content_h)
        
        return None
    
    def _detect_by_color_segmentation(self, image: np.ndarray, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect content by segmenting uniform border color from varied content.
        Works for both bright and colored borders.
        """
        h, w = gray.shape
        
        # Convert to LAB for better color segmentation
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Sample border color (average of all borders)
            border_size = min(h, w) // 10
            border_colors = []
            
            # Sample from borders
            for region in [lab[:border_size, :], lab[-border_size:, :],
                          lab[:, :border_size], lab[:, -border_size:]]:
                if region.size > 0:
                    border_colors.append(np.mean(region.reshape(-1, 3), axis=0))
            
            if border_colors:
                avg_border_color = np.mean(border_colors, axis=0)
                
                # Calculate distance from border color for each pixel
                diff = np.sqrt(np.sum((lab.astype(np.float32) - avg_border_color) ** 2, axis=2))
                
                # Threshold to separate content from border
                threshold = np.percentile(diff, 30)  # Border should be similar to avg
                content_mask = (diff > threshold).astype(np.uint8) * 255
            else:
                content_mask = None
        else:
            # Grayscale fallback
            border_size = min(h, w) // 10
            border_mean = np.mean([
                np.mean(gray[:border_size, :]),
                np.mean(gray[-border_size:, :]),
                np.mean(gray[:, :border_size]),
                np.mean(gray[:, -border_size:])
            ])
            
            diff = np.abs(gray.astype(np.float32) - border_mean)
            threshold = 30
            content_mask = (diff > threshold).astype(np.uint8) * 255
        
        if content_mask is None:
            return None
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)
        content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest centered contour
            center = np.array([w // 2, h // 2])
            best_contour = None
            best_score = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < h * w * 0.1:  # Too small
                    continue
                
                x, y, cw, ch = cv2.boundingRect(cnt)
                cx, cy = x + cw // 2, y + ch // 2
                
                # Score based on area and centrality
                dist = np.linalg.norm(np.array([cx, cy]) - center)
                centrality = 1 - (dist / (max(h, w) * 0.5))
                score = area * centrality
                
                if score > best_score:
                    best_score = score
                    best_contour = cnt
            
            if best_contour is not None:
                x, y, cw, ch = cv2.boundingRect(best_contour)
                margin = self.config["margin_pixels"]
                x = max(0, x - margin)
                y = max(0, y - margin)
                cw = min(w - x, cw + 2 * margin)
                ch = min(h - y, ch + 2 * margin)
                
                return (x, y, cw, ch)
        
        return None
    
    def _detect_by_rectangular_contour(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect rectangular content region using contour detection.
        Looks for rectangular shapes inside the image borders.
        """
        h, w = gray.shape
        
        # Multi-scale edge detection
        for canny_low, canny_high in [(30, 80), (50, 150), (80, 200)]:
            edges = cv2.Canny(gray, canny_low, canny_high)
            
            # Dilate to connect broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < h * w * 0.15 or area > h * w * 0.95:
                    continue
                
                # Approximate to polygon
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                # Check if it's roughly rectangular (4 vertices)
                if 4 <= len(approx) <= 6:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    
                    # Check aspect ratio (not too extreme)
                    aspect = cw / ch if ch > 0 else 0
                    if 0.3 < aspect < 3.0:
                        # Check if it's inside the image (not touching borders)
                        border = min(h, w) // 20
                        if x > border and y > border and x + cw < w - border and y + ch < h - border:
                            margin = self.config["margin_pixels"]
                            x = max(0, x - margin)
                            y = max(0, y - margin)
                            cw = min(w - x, cw + 2 * margin)
                            ch = min(h - y, ch + 2 * margin)
                            
                            return (x, y, cw, ch)
        
        return None
    
    def detect_polaroid_content(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        """
        Detect the actual photo content inside a polaroid frame using multiple strategies.
        
        PRIORITY ORDER:
        1. Strict white border detection (best for actual polaroids)
        2. White border with gradient analysis (handles less uniform whites)
        3. Fallback methods (color segmentation, contour, etc.)
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (crop coordinates, detection method) or (None, "none")
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # FIRST: Try strict white border detection (optimized for polaroids)
        if self.config.get("strict_white_detection", True):
            # Method 1: Strict white border detection
            try:
                result = self._detect_strict_white_borders(image, gray)
                if result is not None:
                    x, y, cw, ch = result
                    if cw > w * 0.25 and ch > h * 0.25:
                        return result, "strict_white_border"
            except Exception:
                pass
            
            # Method 2: White border with gradient analysis
            try:
                result = self._detect_white_border_with_gradient(image, gray)
                if result is not None:
                    x, y, cw, ch = result
                    if cw > w * 0.25 and ch > h * 0.25:
                        return result, "white_border_gradient"
            except Exception:
                pass
        
        # Analyze image properties for fallback methods
        analysis = self._analyze_region_properties(image, gray)
        
        # If no clear frame structure, might not be a polaroid
        if not analysis['has_frame_structure'] and analysis['variance_ratio'] < 1.2:
            # Still try detection methods but with lower confidence
            pass
        
        # FALLBACK: Try other detection methods in order of reliability
        detection_methods = [
            (self._detect_by_color_segmentation, "color_segmentation"),
            (self._detect_by_rectangular_contour, "rectangular_contour"),
            (self._detect_by_gradient_transition, "gradient_transition"),
            (self._detect_by_edge_density, "edge_density"),
        ]
        
        results = []
        
        for method_func, method_name in detection_methods:
            try:
                if method_name == "color_segmentation":
                    result = method_func(image, gray)
                else:
                    result = method_func(gray)
                
                if result is not None:
                    x, y, cw, ch = result
                    # Validate result
                    if cw > w * 0.25 and ch > h * 0.25 and cw < w * 0.98 and ch < h * 0.98:
                        # Score based on how "centered" and "reasonably sized" it is
                        center_x, center_y = x + cw // 2, y + ch // 2
                        offset = abs(center_x - w // 2) + abs(center_y - h // 2)
                        area_ratio = (cw * ch) / (w * h)
                        
                        # Prefer larger, centered results
                        score = area_ratio * (1 - offset / (w + h))
                        results.append((result, method_name, score))
            except Exception:
                continue
        
        if not results:
            return None, "none"
        
        # Return best result
        best = max(results, key=lambda x: x[2])
        return best[0], best[1]
    
    def process(self, image: np.ndarray) -> PolaroidCropResult:
        """
        Extract photo content from polaroid frame.
        
        Args:
            image: BGR input image
            
        Returns:
            PolaroidCropResult with cropped image and metadata
        """
        import time
        start_time = time.time()
        
        h, w = image.shape[:2]
        original_area = h * w
        
        # Detect content region
        crop_coords, method = self.detect_polaroid_content(image)
        
        if crop_coords is None:
            return PolaroidCropResult(
                image=image.copy(),
                crop_region=None,
                border_detected=False,
                content_ratio=1.0,
                detection_method="none",
                processing_time=time.time() - start_time
            )
        
        x, y, w_crop, h_crop = crop_coords
        cropped = image[y:y+h_crop, x:x+w_crop]
        
        content_ratio = (w_crop * h_crop) / original_area
        
        return PolaroidCropResult(
            image=cropped,
            crop_region=crop_coords,
            border_detected=True,
            content_ratio=content_ratio,
            detection_method=method,
            processing_time=time.time() - start_time
        )


def crop_polaroid_content(image: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
    """
    Convenience function to crop polaroid content.
    
    Args:
        image: BGR input image
        config: Optional configuration
        
    Returns:
        Cropped image with just the photo content
    """
    cropper = PolaroidCropper(config)
    result = cropper.process(image)
    return result.image

