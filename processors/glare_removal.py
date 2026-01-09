"""
Glare detection and removal module for laminated/polaroid photographs.
Implements multiple strategies for removing specular highlights and reflections.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GlareResult:
    """Result from glare removal processing."""
    image: np.ndarray
    mask: Optional[np.ndarray]
    regions_detected: int
    coverage_percent: float
    method_used: str
    processing_time: float
    quality_preserved: bool


class GlareRemover:
    """
    Multi-strategy glare removal for digitized photographs.
    
    Supports:
    - Adaptive thresholding with inpainting
    - Specular highlight detection
    - Multi-exposure blending simulation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize glare remover with configuration.
        
        Args:
            config: Glare removal configuration dictionary
        """
        self.config = config or {}
        self.default_config = {
            "detection_threshold": "auto",
            "method": "adaptive",
            "inpaint_radius": 5,
            "highlight_percentile": 98,
            "min_glare_area": 100,
            "blend_factor": 0.7,
        }
        # Merge with defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def detect_glare_regions(self, image: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Detect glare/specular highlight regions in the image.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (glare_mask, severity_score, num_regions)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Also analyze saturation (glare typically has low saturation)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Calculate threshold
        if self.config["detection_threshold"] == "auto":
            threshold = self._calculate_auto_threshold(l_channel)
        else:
            threshold = self.config["detection_threshold"]
        
        # Create initial mask based on brightness
        brightness_mask = l_channel > threshold
        
        # Refine with saturation (glare has high L, low S)
        low_saturation = saturation < 50
        
        # Combine: bright AND (low saturation OR very bright)
        very_bright = l_channel > 250
        glare_mask = (brightness_mask & low_saturation) | very_bright
        
        # Convert to uint8
        glare_mask = glare_mask.astype(np.uint8) * 255
        
        # Clean up mask - remove small noise regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours to count regions and filter by size
        contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small regions
        min_area = self.config["min_glare_area"]
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        # Create cleaned mask
        cleaned_mask = np.zeros_like(glare_mask)
        cv2.drawContours(cleaned_mask, valid_contours, -1, 255, -1)
        
        # Dilate slightly to capture glare edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=1)
        
        # Calculate severity
        coverage = np.sum(cleaned_mask > 0) / cleaned_mask.size
        severity = min(coverage * 10, 1.0)  # Scale coverage to severity
        
        return cleaned_mask, severity, len(valid_contours)
    
    def _calculate_auto_threshold(self, l_channel: np.ndarray) -> int:
        """
        Automatically calculate glare threshold based on histogram analysis.
        
        Args:
            l_channel: Lightness channel from LAB
            
        Returns:
            Optimal threshold value
        """
        # Calculate percentiles
        p90 = np.percentile(l_channel, 90)
        p95 = np.percentile(l_channel, 95)
        p99 = np.percentile(l_channel, 99)
        
        # If there's a big jump in top percentiles, there's likely glare
        if (p99 - p90) > 20:
            # Set threshold between p90 and p95
            threshold = int(p90 + 0.7 * (p95 - p90))
        else:
            # Conservative threshold
            threshold = int(p99)
        
        # Ensure minimum threshold to avoid false positives
        threshold = max(threshold, 230)
        
        return threshold
    
    def remove_glare_inpainting(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove glare using inpainting technique.
        
        Args:
            image: BGR input image
            mask: Binary mask of glare regions
            
        Returns:
            Image with glare regions inpainted
        """
        radius = self.config["inpaint_radius"]
        
        # Use Telea's method (generally better for photo restoration)
        result = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
        
        # Optional: blend with original for more natural result
        blend_factor = self.config["blend_factor"]
        
        # Create soft mask for blending
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (21, 21), 0)
        soft_mask = np.stack([soft_mask] * 3, axis=-1)
        
        # Blend: result in glare areas, original elsewhere
        blended = (result * soft_mask + image * (1 - soft_mask)).astype(np.uint8)
        
        return blended
    
    def remove_glare_adaptive(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove glare using adaptive local histogram matching.
        
        Args:
            image: BGR input image
            mask: Binary mask of glare regions
            
        Returns:
            Image with glare reduced
        """
        result = image.copy()
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Create soft mask
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (31, 31), 0)
        
        # Calculate local statistics in non-glare regions
        non_glare_mask = (mask == 0)
        
        if np.sum(non_glare_mask) > 0:
            # Get statistics from non-glare areas
            non_glare_mean = np.mean(l_channel[non_glare_mask])
            non_glare_std = np.std(l_channel[non_glare_mask])
            
            # For glare pixels, reduce brightness towards local mean
            glare_pixels = soft_mask > 0.1
            
            # Calculate correction factor
            correction = np.zeros_like(l_channel)
            correction[glare_pixels] = (l_channel[glare_pixels] - non_glare_mean) * soft_mask[glare_pixels] * 0.7
            
            # Apply correction
            l_channel_corrected = l_channel - correction
            l_channel_corrected = np.clip(l_channel_corrected, 0, 255).astype(np.uint8)
            
            # Reconstruct image
            lab[:, :, 0] = l_channel_corrected
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def remove_glare_guided_filter(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove glare using guided filter to estimate underlying texture.
        
        Args:
            image: BGR input image
            mask: Binary mask of glare regions
            
        Returns:
            Image with glare regions replaced by filtered estimate
        """
        # Use guided filter to estimate underlying texture
        radius = 15
        eps = 1000
        
        # Apply guided filter
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Process each channel
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            # Simplified guided filter using bilateral filter as approximation
            filtered = cv2.bilateralFilter(channel.astype(np.uint8), radius, 75, 75)
            result[:, :, i] = filtered
        
        # Create soft mask for blending
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (21, 21), 0)
        soft_mask = np.stack([soft_mask] * 3, axis=-1)
        
        # Blend filtered result with original
        blended = (result * soft_mask + image * (1 - soft_mask)).astype(np.uint8)
        
        return blended
    
    def _check_quality(self, original: np.ndarray, processed: np.ndarray) -> bool:
        """
        Check if processing preserved quality.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            True if quality is preserved
        """
        # Simple check: processed should not have drastically different statistics
        orig_mean = np.mean(original)
        proc_mean = np.mean(processed)
        
        # Mean shouldn't change dramatically
        if abs(orig_mean - proc_mean) > 30:
            return False
        
        # Check for artifacts (large local differences in non-glare regions)
        diff = cv2.absdiff(original, processed)
        max_diff = np.max(diff)
        
        # Large max differences might indicate artifacts
        if max_diff > 200:
            return False
        
        return True
    
    def process(self, image: np.ndarray) -> GlareResult:
        """
        Main processing function for glare removal.
        
        Args:
            image: BGR input image
            
        Returns:
            GlareResult with processed image and metadata
        """
        import time
        start_time = time.time()
        
        # Detect glare regions
        mask, severity, num_regions = self.detect_glare_regions(image)
        
        # Calculate coverage
        coverage = np.sum(mask > 0) / mask.size * 100
        
        # If no significant glare, return original
        if severity < 0.05 or num_regions == 0:
            return GlareResult(
                image=image.copy(),
                mask=None,
                regions_detected=0,
                coverage_percent=0.0,
                method_used="none",
                processing_time=time.time() - start_time,
                quality_preserved=True
            )
        
        # Choose method based on config or severity
        method = self.config["method"]
        if method == "auto":
            method = "inpainting" if severity > 0.3 else "adaptive"
        
        # Apply chosen method
        if method == "inpainting":
            processed = self.remove_glare_inpainting(image, mask)
        elif method == "adaptive":
            processed = self.remove_glare_adaptive(image, mask)
        elif method == "guided":
            processed = self.remove_glare_guided_filter(image, mask)
        else:
            processed = self.remove_glare_inpainting(image, mask)
        
        # Quality check
        quality_preserved = self._check_quality(image, processed)
        
        # If quality degraded, try alternative method or return original
        if not quality_preserved:
            # Try adaptive method as fallback
            if method != "adaptive":
                processed = self.remove_glare_adaptive(image, mask)
                quality_preserved = self._check_quality(image, processed)
                method = "adaptive (fallback)"
            
            # If still bad, return original
            if not quality_preserved:
                return GlareResult(
                    image=image.copy(),
                    mask=mask,
                    regions_detected=num_regions,
                    coverage_percent=coverage,
                    method_used="none (quality check failed)",
                    processing_time=time.time() - start_time,
                    quality_preserved=False
                )
        
        return GlareResult(
            image=processed,
            mask=mask,
            regions_detected=num_regions,
            coverage_percent=coverage,
            method_used=method,
            processing_time=time.time() - start_time,
            quality_preserved=quality_preserved
        )


def detect_glare_regions(image: np.ndarray, config: Optional[Dict] = None) -> Tuple[np.ndarray, float]:
    """
    Convenience function to detect glare regions.
    
    Args:
        image: BGR input image
        config: Optional configuration
        
    Returns:
        Tuple of (glare_mask, severity_score)
    """
    remover = GlareRemover(config)
    mask, severity, _ = remover.detect_glare_regions(image)
    return mask, severity


def remove_glare(image: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
    """
    Convenience function to remove glare from image.
    
    Args:
        image: BGR input image
        config: Optional configuration
        
    Returns:
        Processed image with glare removed
    """
    remover = GlareRemover(config)
    result = remover.process(image)
    return result.image

