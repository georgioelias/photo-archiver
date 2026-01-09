"""
Color correction module for faded/discolored photographs.
Implements white balance, histogram equalization, and color cast removal.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ColorCorrectionResult:
    """Result from color correction processing."""
    image: np.ndarray
    white_balance_method: str
    color_cast_detected: Tuple[float, float, float]
    color_cast_corrected: bool
    saturation_adjusted: bool
    histogram_method: str
    processing_time: float


class ColorCorrector:
    """
    Comprehensive color correction for digitized photographs.
    
    Supports:
    - Multiple white balance algorithms
    - CLAHE and standard histogram equalization
    - Color cast detection and removal
    - Saturation boosting for faded photos
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize color corrector with configuration.
        
        Args:
            config: Color correction configuration dictionary
        """
        self.config = config or {}
        self.default_config = {
            "white_balance": "auto",
            "histogram_method": "clahe",
            "clahe_clip_limit": 2.0,
            "clahe_grid_size": (8, 8),
            "color_cast_removal": True,
            "saturation_boost": 1.1,
            "gamma": 1.0,
            "preserve_skin_tones": True,
        }
        # Merge with defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def white_balance_gray_world(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gray world white balance algorithm.
        Assumes the average color of the scene should be gray.
        
        Args:
            image: BGR input image
            
        Returns:
            White-balanced image
        """
        result = image.copy().astype(np.float32)
        
        # Calculate mean of each channel
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        
        # Calculate overall average
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        # Scale factors
        scale_b = avg_gray / avg_b if avg_b > 0 else 1.0
        scale_g = avg_gray / avg_g if avg_g > 0 else 1.0
        scale_r = avg_gray / avg_r if avg_r > 0 else 1.0
        
        # Limit scaling to prevent extreme adjustments
        max_scale = 2.0
        scale_b = min(scale_b, max_scale)
        scale_g = min(scale_g, max_scale)
        scale_r = min(scale_r, max_scale)
        
        # Apply scaling
        result[:, :, 0] *= scale_b
        result[:, :, 1] *= scale_g
        result[:, :, 2] *= scale_r
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def white_balance_white_patch(self, image: np.ndarray) -> np.ndarray:
        """
        Apply white patch white balance algorithm.
        Uses the brightest region as white reference.
        
        Args:
            image: BGR input image
            
        Returns:
            White-balanced image
        """
        result = image.copy().astype(np.float32)
        
        # Find the brightest pixels (top 1%)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, 99)
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) < 10:
            # Not enough bright pixels, use top 5%
            threshold = np.percentile(gray, 95)
            bright_mask = gray >= threshold
        
        # Get average color of bright regions
        if np.sum(bright_mask) > 0:
            white_b = np.mean(result[:, :, 0][bright_mask])
            white_g = np.mean(result[:, :, 1][bright_mask])
            white_r = np.mean(result[:, :, 2][bright_mask])
            
            # Calculate scale to make white regions neutral
            max_val = max(white_b, white_g, white_r)
            
            if max_val > 0:
                scale_b = max_val / white_b if white_b > 0 else 1.0
                scale_g = max_val / white_g if white_g > 0 else 1.0
                scale_r = max_val / white_r if white_r > 0 else 1.0
                
                # Limit scaling
                max_scale = 2.0
                scale_b = min(scale_b, max_scale)
                scale_g = min(scale_g, max_scale)
                scale_r = min(scale_r, max_scale)
                
                # Apply scaling
                result[:, :, 0] *= scale_b
                result[:, :, 1] *= scale_g
                result[:, :, 2] *= scale_r
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def white_balance_combined(self, image: np.ndarray, 
                               gray_weight: float = 0.5) -> np.ndarray:
        """
        Apply combined white balance using both gray world and white patch.
        
        Args:
            image: BGR input image
            gray_weight: Weight for gray world (0-1), white patch gets 1-gray_weight
            
        Returns:
            White-balanced image
        """
        gray_result = self.white_balance_gray_world(image)
        white_result = self.white_balance_white_patch(image)
        
        # Weighted blend
        combined = cv2.addWeighted(
            gray_result, gray_weight,
            white_result, 1.0 - gray_weight,
            0
        )
        
        return combined
    
    def auto_white_balance(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Automatically select and apply best white balance method.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (white-balanced image, method used)
        """
        # Analyze image to determine best method
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for presence of bright/white regions
        bright_threshold = np.percentile(gray, 95)
        has_bright_regions = bright_threshold > 200
        
        # Check color distribution
        means = np.mean(image, axis=(0, 1))
        color_variance = np.std(means)
        
        if has_bright_regions and color_variance > 20:
            # Has white reference points - use white patch
            return self.white_balance_white_patch(image), "white_patch"
        elif color_variance > 30:
            # Strong color cast - use gray world
            return self.white_balance_gray_world(image), "gray_world"
        else:
            # Use combined approach
            return self.white_balance_combined(image), "combined"
    
    def detect_color_cast(self, image: np.ndarray) -> Tuple[float, float, float]:
        """
        Detect color cast in the image.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (B, G, R) deviation from neutral
        """
        # Calculate mean of each channel
        means = np.mean(image, axis=(0, 1))
        overall_mean = np.mean(means)
        
        # Deviation from neutral gray
        deviations = (means - overall_mean) / 255.0
        
        return tuple(float(d) for d in deviations)
    
    def remove_color_cast(self, image: np.ndarray) -> np.ndarray:
        """
        Remove detected color cast from the image.
        
        Args:
            image: BGR input image
            
        Returns:
            Color-corrected image
        """
        result = image.copy().astype(np.float32)
        
        # Detect cast
        cast = self.detect_color_cast(image)
        
        # Only correct if cast is significant
        if max(abs(c) for c in cast) < 0.03:
            return image
        
        # Calculate correction
        correction_strength = 0.7  # Don't fully correct to preserve some character
        
        # Apply per-channel correction
        means = np.mean(result, axis=(0, 1))
        target_mean = np.mean(means)
        
        for i in range(3):
            if means[i] != target_mean:
                scale = 1.0 + (target_mean - means[i]) / means[i] * correction_strength
                scale = np.clip(scale, 0.5, 2.0)
                result[:, :, i] *= scale
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Applied to L channel only to preserve colors.
        
        Args:
            image: BGR input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Create CLAHE object
        clip_limit = self.config["clahe_clip_limit"]
        grid_size = self.config["clahe_grid_size"]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        
        # Apply to L channel
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def apply_histogram_equalization(self, image: np.ndarray, 
                                      method: str = "clahe") -> np.ndarray:
        """
        Apply histogram equalization using specified method.
        
        Args:
            image: BGR input image
            method: "clahe", "standard", or "adaptive"
            
        Returns:
            Enhanced image
        """
        if method == "clahe":
            return self.apply_clahe(image)
        
        elif method == "standard":
            # Standard histogram equalization on L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        elif method == "adaptive":
            # Adaptive based on image characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray) / 127.5
            
            if contrast < 0.3:
                # Low contrast - use standard
                return self.apply_histogram_equalization(image, "standard")
            else:
                # Normal/high contrast - use CLAHE
                return self.apply_clahe(image)
        
        return image
    
    def adjust_saturation(self, image: np.ndarray, factor: float = 1.1) -> np.ndarray:
        """
        Adjust image saturation.
        
        Args:
            image: BGR input image
            factor: Saturation multiplier (>1 increases, <1 decreases)
            
        Returns:
            Saturation-adjusted image
        """
        if abs(factor - 1.0) < 0.01:
            return image
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation
        hsv[:, :, 1] *= factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction to adjust brightness.
        
        Args:
            image: BGR input image
            gamma: Gamma value (<1 brightens, >1 darkens)
            
        Returns:
            Gamma-corrected image
        """
        if abs(gamma - 1.0) < 0.01:
            return image
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply lookup table
        return cv2.LUT(image, table)
    
    def detect_skin_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect skin-tone regions in the image.
        
        Args:
            image: BGR input image
            
        Returns:
            Binary mask of skin regions
        """
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Skin detection thresholds
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def process(self, image: np.ndarray) -> ColorCorrectionResult:
        """
        Main processing function for color correction.
        
        Args:
            image: BGR input image
            
        Returns:
            ColorCorrectionResult with corrected image and metadata
        """
        import time
        start_time = time.time()
        
        result = image.copy()
        
        # Detect initial color cast
        color_cast = self.detect_color_cast(image)
        color_cast_corrected = False
        
        # Apply white balance
        wb_method = self.config.get("white_balance", "auto")
        wb_method_used = wb_method
        
        if wb_method == "auto":
            result, wb_method_used = self.auto_white_balance(result)
        elif wb_method == "gray_world":
            result = self.white_balance_gray_world(result)
        elif wb_method == "white_patch":
            result = self.white_balance_white_patch(result)
        elif wb_method == "combined":
            result = self.white_balance_combined(result)
        # else: no white balance applied
        
        # Remove color cast if enabled
        if self.config.get("color_cast_removal", True):
            if max(abs(c) for c in color_cast) > 0.05:
                result = self.remove_color_cast(result)
                color_cast_corrected = True
        
        # Apply histogram equalization
        hist_method = self.config.get("histogram_method", "clahe")
        if hist_method != "off":
            result = self.apply_histogram_equalization(result, hist_method)
        
        # Apply gamma correction if specified
        gamma = self.config.get("gamma", 1.0)
        if gamma != 1.0:
            result = self.apply_gamma_correction(result, gamma)
        
        # Adjust saturation
        saturation_boost = self.config.get("saturation_boost", 1.0)
        saturation_adjusted = False
        
        if saturation_boost != 1.0:
            # If preserving skin tones, apply differently
            if self.config.get("preserve_skin_tones", True):
                skin_mask = self.detect_skin_regions(result)
                
                # Apply full saturation boost to non-skin areas
                saturated = self.adjust_saturation(result, saturation_boost)
                
                # Blend based on skin mask (less boost on skin)
                skin_factor = 0.5  # 50% of the boost on skin
                skin_saturated = self.adjust_saturation(result, 1.0 + (saturation_boost - 1.0) * skin_factor)
                
                # Combine
                skin_mask_3d = np.stack([skin_mask / 255.0] * 3, axis=-1)
                result = (saturated * (1 - skin_mask_3d) + skin_saturated * skin_mask_3d).astype(np.uint8)
            else:
                result = self.adjust_saturation(result, saturation_boost)
            
            saturation_adjusted = True
        
        return ColorCorrectionResult(
            image=result,
            white_balance_method=wb_method_used,
            color_cast_detected=color_cast,
            color_cast_corrected=color_cast_corrected,
            saturation_adjusted=saturation_adjusted,
            histogram_method=hist_method,
            processing_time=time.time() - start_time
        )


def correct_colors(image: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
    """
    Convenience function to correct colors.
    
    Args:
        image: BGR input image
        config: Optional configuration
        
    Returns:
        Color-corrected image
    """
    corrector = ColorCorrector(config)
    result = corrector.process(image)
    return result.image

