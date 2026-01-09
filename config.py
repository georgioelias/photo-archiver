"""
Configuration system for Digital Image Archiving System.
Provides auto-configuration with sensible defaults that work for typical laminated photos.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# Default configuration - works out-of-the-box for most photos
DEFAULT_CONFIG = {
    # Glare Removal
    "glare": {
        "detection_threshold": "auto",
        "method": "adaptive",  # adaptive | inpainting | multi-exposure-blend
        "inpaint_radius": 5,
        "highlight_percentile": 98,
        "min_glare_area": 100,  # Minimum pixels to consider as glare
        "blend_factor": 0.7,
    },
    
    # Perspective Correction
    "perspective": {
        "border_detection": "auto",
        "method": "contour",  # contour | hough | corner
        "margin_percent": 2,
        "min_area_ratio": 0.3,
        "canny_low": 50,
        "canny_high": 150,
        "blur_kernel": 5,
        "epsilon_factor": 0.02,
    },
    
    # Color Correction
    "color": {
        "white_balance": "auto",  # auto | gray_world | white_patch | manual
        "histogram_method": "clahe",  # clahe | standard | adaptive
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": (8, 8),
        "color_cast_removal": True,
        "saturation_boost": 1.1,
        "gamma_correction": "auto",
    },
    
    # Enhancement
    "enhancement": {
        "denoise": "auto",  # auto | off | light | medium | strong
        "denoise_strength": "auto",
        "sharpen": "auto",  # auto | off | light | medium | strong
        "unsharp_amount": 1.0,
        "unsharp_radius": 1.0,
        "contrast": "auto",  # auto | off | stretch | normalize
    },
    
    # Compression targets (in KB)
    "compression_targets": [30, 100, 500, 1000],
    
    # Quality thresholds
    "quality": {
        "min_psnr": 25.0,  # Minimum acceptable PSNR
        "min_ssim": 0.85,  # Minimum acceptable SSIM
        "degradation_threshold": 0.1,  # Max allowed quality loss
    }
}


@dataclass
class ImageAnalysis:
    """Results from comprehensive image analysis."""
    brightness: float = 0.5
    contrast: float = 0.5
    color_cast: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise_level: float = 0.0
    has_glare: bool = False
    glare_severity: float = 0.0
    glare_regions: Optional[np.ndarray] = None
    sharpness: float = 0.5
    has_border: bool = False
    is_faded: bool = False
    dominant_colors: list = field(default_factory=list)
    histogram_stats: Dict[str, Any] = field(default_factory=dict)


class AutoConfig:
    """Automatically determines optimal processing parameters based on image analysis."""
    
    def __init__(self, image: np.ndarray):
        """
        Initialize auto-configuration with an image.
        
        Args:
            image: BGR numpy array
        """
        self.image = image
        self.analysis = self._analyze_image()
    
    def _analyze_image(self) -> ImageAnalysis:
        """Perform comprehensive image analysis."""
        analysis = ImageAnalysis()
        
        # Analyze brightness
        analysis.brightness = self._analyze_brightness()
        
        # Analyze contrast
        analysis.contrast = self._analyze_contrast()
        
        # Detect color cast
        analysis.color_cast = self._detect_color_cast()
        
        # Estimate noise
        analysis.noise_level = self._estimate_noise()
        
        # Detect glare
        analysis.has_glare, analysis.glare_severity, analysis.glare_regions = self._detect_glare()
        
        # Measure sharpness
        analysis.sharpness = self._measure_sharpness()
        
        # Detect border
        analysis.has_border = self._detect_border()
        
        # Detect fading
        analysis.is_faded = self._detect_fading()
        
        # Get histogram statistics
        analysis.histogram_stats = self._get_histogram_stats()
        
        return analysis
    
    def _analyze_brightness(self) -> float:
        """
        Analyze overall image brightness.
        
        Returns:
            Float 0-1, where 0 is very dark and 1 is very bright
        """
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        mean_brightness = np.mean(gray) / 255.0
        return float(mean_brightness)
    
    def _analyze_contrast(self) -> float:
        """
        Analyze image contrast using standard deviation.
        
        Returns:
            Float 0-1, normalized contrast value
        """
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        # Use standard deviation as contrast measure
        std_dev = np.std(gray) / 127.5  # Normalize to roughly 0-1
        return float(min(std_dev, 1.0))
    
    def _detect_color_cast(self) -> Tuple[float, float, float]:
        """
        Detect color cast in the image.
        
        Returns:
            Tuple of (red, green, blue) deviations from neutral
        """
        if len(self.image.shape) != 3:
            return (0.0, 0.0, 0.0)
        
        # Calculate mean of each channel
        means = np.mean(self.image, axis=(0, 1))
        overall_mean = np.mean(means)
        
        # Calculate deviation from neutral (gray)
        deviations = (means - overall_mean) / 255.0
        
        return tuple(float(d) for d in deviations)
    
    def _estimate_noise(self) -> float:
        """
        Estimate noise level using Laplacian variance method.
        
        Returns:
            Float 0-1 representing noise level
        """
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        # Laplacian variance method for noise estimation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize - typical values range from 100 (clean) to 1000+ (noisy)
        # Map to 0-1 scale
        noise_level = min(variance / 1000.0, 1.0)
        return float(noise_level)
    
    def _detect_glare(self) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect glare/specular highlights in the image.
        
        Returns:
            Tuple of (has_glare, severity 0-1, glare_mask)
        """
        # Convert to LAB for better highlight detection
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate percentiles
        p90 = np.percentile(l_channel, 90)
        p99 = np.percentile(l_channel, 99)
        p100 = np.percentile(l_channel, 100)
        
        # Glare detection: significant jump in top percentiles
        has_glare = (p99 - p90) > 30 and p99 > 230
        
        if not has_glare:
            return False, 0.0, None
        
        # Create glare mask
        threshold = p90 + 0.5 * (p99 - p90)
        glare_mask = (l_channel > threshold).astype(np.uint8) * 255
        
        # Calculate severity based on glare area
        glare_area = np.sum(glare_mask > 0) / glare_mask.size
        severity = min(glare_area * 10, 1.0)  # Scale up, cap at 1
        
        # Also consider brightness intensity
        intensity_factor = (p100 - p99) / 25.0
        severity = min(severity + intensity_factor * 0.3, 1.0)
        
        return True, float(severity), glare_mask
    
    def _measure_sharpness(self) -> float:
        """
        Measure image sharpness using Laplacian variance.
        
        Returns:
            Float 0-1 representing sharpness
        """
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        # Laplacian variance as sharpness measure
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize - sharp images typically have variance > 500
        normalized = min(sharpness / 1000.0, 1.0)
        return float(normalized)
    
    def _detect_border(self) -> bool:
        """
        Detect if image has a distinct border (e.g., polaroid white border).
        
        Returns:
            Boolean indicating presence of border
        """
        # Sample edges of the image
        h, w = self.image.shape[:2]
        border_size = min(h, w) // 20
        
        # Get border regions
        top = self.image[:border_size, :]
        bottom = self.image[-border_size:, :]
        left = self.image[:, :border_size]
        right = self.image[:, -border_size:]
        
        # Calculate mean brightness of borders
        border_means = []
        for region in [top, bottom, left, right]:
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            border_means.append(np.mean(gray))
        
        avg_border = np.mean(border_means)
        
        # Calculate center brightness
        center_y, center_x = h // 2, w // 2
        center_size = min(h, w) // 4
        center = self.image[center_y-center_size:center_y+center_size,
                           center_x-center_size:center_x+center_size]
        if len(center.shape) == 3:
            center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        else:
            center_gray = center
        center_mean = np.mean(center_gray)
        
        # Border detected if edges are significantly brighter than center
        has_border = avg_border > center_mean + 30 and avg_border > 200
        
        return has_border
    
    def _detect_fading(self) -> bool:
        """
        Detect if the photo appears faded (low saturation, compressed histogram).
        
        Returns:
            Boolean indicating if photo is faded
        """
        if len(self.image.shape) != 3:
            return False
        
        # Convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Low average saturation indicates fading
        avg_saturation = np.mean(saturation)
        
        # Also check histogram spread
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)
        dynamic_range = p95 - p5
        
        # Faded if low saturation and compressed dynamic range
        is_faded = avg_saturation < 60 or dynamic_range < 150
        
        return is_faded
    
    def _get_histogram_stats(self) -> Dict[str, Any]:
        """
        Calculate histogram statistics for the image.
        
        Returns:
            Dictionary of histogram statistics
        """
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Calculate statistics
        values = np.arange(256)
        total_pixels = np.sum(hist)
        
        mean = np.sum(values * hist) / total_pixels
        variance = np.sum(((values - mean) ** 2) * hist) / total_pixels
        
        # Find mode (most common value)
        mode = np.argmax(hist)
        
        # Calculate percentiles
        cumsum = np.cumsum(hist) / total_pixels
        p1 = np.searchsorted(cumsum, 0.01)
        p5 = np.searchsorted(cumsum, 0.05)
        p50 = np.searchsorted(cumsum, 0.50)
        p95 = np.searchsorted(cumsum, 0.95)
        p99 = np.searchsorted(cumsum, 0.99)
        
        return {
            "mean": float(mean),
            "variance": float(variance),
            "std_dev": float(np.sqrt(variance)),
            "mode": int(mode),
            "p1": int(p1),
            "p5": int(p5),
            "p50": int(p50),
            "p95": int(p95),
            "p99": int(p99),
            "dynamic_range": int(p99 - p1),
        }
    
    def get_glare_config(self) -> Dict[str, Any]:
        """Return optimal glare removal parameters based on analysis."""
        if not self.analysis.has_glare:
            return {"enabled": False}
        
        severity = self.analysis.glare_severity
        
        # Choose method based on severity
        if severity > 0.5:
            method = "inpainting"
        elif severity > 0.2:
            method = "adaptive"
        else:
            method = "adaptive"
        
        # Calculate inpaint radius based on severity
        inpaint_radius = 3 + int(severity * 7)  # 3-10 based on severity
        
        # Calculate threshold
        stats = self.analysis.histogram_stats
        threshold = stats.get("p95", 230) + int((255 - stats.get("p95", 230)) * 0.5)
        
        return {
            "enabled": True,
            "method": method,
            "inpaint_radius": inpaint_radius,
            "threshold": min(threshold, 250),
            "blend_factor": 0.8 if severity > 0.3 else 0.6,
        }
    
    def get_perspective_config(self) -> Dict[str, Any]:
        """Return optimal perspective correction parameters."""
        return {
            "enabled": True,
            "method": "contour",
            "margin_percent": 2 if self.analysis.has_border else 1,
            "detect_inner_photo": self.analysis.has_border,
            "min_area_ratio": 0.3,
        }
    
    def get_color_config(self) -> Dict[str, Any]:
        """Return optimal color correction parameters based on analysis."""
        # Choose white balance method
        if self.analysis.has_border:
            wb_method = "white_patch"
        elif max(abs(c) for c in self.analysis.color_cast) > 0.1:
            wb_method = "gray_world"
        else:
            wb_method = "combined"
        
        # Adjust saturation boost for faded photos
        saturation_boost = 1.2 if self.analysis.is_faded else 1.05
        
        # Adjust CLAHE parameters based on contrast
        if self.analysis.contrast < 0.3:
            clahe_clip = 3.0
        elif self.analysis.contrast > 0.7:
            clahe_clip = 1.5
        else:
            clahe_clip = 2.0
        
        # Gamma correction
        if self.analysis.brightness < 0.4:
            gamma = 0.8  # Brighten
        elif self.analysis.brightness > 0.6:
            gamma = 1.2  # Darken slightly
        else:
            gamma = 1.0
        
        return {
            "enabled": True,
            "white_balance_method": wb_method,
            "saturation_boost": saturation_boost,
            "clahe_clip_limit": clahe_clip,
            "clahe_grid_size": (8, 8),
            "color_cast_removal": max(abs(c) for c in self.analysis.color_cast) > 0.05,
            "gamma": gamma,
        }
    
    def get_enhancement_config(self) -> Dict[str, Any]:
        """Return optimal enhancement parameters based on analysis."""
        # Denoise strength based on noise level
        noise = self.analysis.noise_level
        if noise < 0.1:
            denoise = "off"
            denoise_strength = 0
        elif noise < 0.3:
            denoise = "light"
            denoise_strength = 3
        elif noise < 0.6:
            denoise = "medium"
            denoise_strength = 6
        else:
            denoise = "strong"
            denoise_strength = 10
        
        # Sharpen based on sharpness level
        sharpness = self.analysis.sharpness
        if sharpness > 0.7:
            sharpen = "off"
            unsharp_amount = 0
        elif sharpness > 0.4:
            sharpen = "light"
            unsharp_amount = 0.5
        elif sharpness > 0.2:
            sharpen = "medium"
            unsharp_amount = 1.0
        else:
            sharpen = "strong"
            unsharp_amount = 1.5
        
        # Contrast enhancement
        contrast = self.analysis.contrast
        if contrast < 0.3:
            contrast_mode = "stretch"
        elif contrast > 0.8:
            contrast_mode = "off"
        else:
            contrast_mode = "normalize"
        
        return {
            "enabled": True,
            "denoise": denoise,
            "denoise_strength": denoise_strength,
            "sharpen": sharpen,
            "unsharp_amount": unsharp_amount,
            "unsharp_radius": 1.0,
            "contrast": contrast_mode,
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """Return complete configuration for all processing steps."""
        return {
            "glare": self.get_glare_config(),
            "perspective": self.get_perspective_config(),
            "color": self.get_color_config(),
            "enhancement": self.get_enhancement_config(),
            "analysis": {
                "brightness": self.analysis.brightness,
                "contrast": self.analysis.contrast,
                "noise_level": self.analysis.noise_level,
                "sharpness": self.analysis.sharpness,
                "has_glare": self.analysis.has_glare,
                "glare_severity": self.analysis.glare_severity,
                "has_border": self.analysis.has_border,
                "is_faded": self.analysis.is_faded,
                "color_cast": self.analysis.color_cast,
            }
        }


def get_default_config() -> Dict[str, Any]:
    """Return a copy of the default configuration."""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override config into base config.
    
    Args:
        base: Base configuration dictionary
        override: Override values to apply
    
    Returns:
        Merged configuration
    """
    import copy
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result

