"""
JPEG compression module with target size optimization.
Implements binary search for achieving specific file sizes.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import io


@dataclass
class CompressionResult:
    """Result from JPEG compression."""
    image_bytes: bytes
    quality: int
    file_size_kb: float
    target_kb: float
    within_tolerance: bool
    iterations: int


class JPEGCompressor:
    """
    JPEG compression with intelligent quality selection.
    
    Features:
    - Binary search for target file size
    - Quality estimation based on image characteristics
    - Batch compression to multiple targets
    """
    
    def __init__(self, tolerance_percent: float = 5.0):
        """
        Initialize compressor.
        
        Args:
            tolerance_percent: Acceptable deviation from target size
        """
        self.tolerance_percent = tolerance_percent
    
    def compress(self, image: np.ndarray, quality: int) -> bytes:
        """
        Compress image to JPEG with specified quality.
        
        Args:
            image: BGR numpy array
            quality: JPEG quality (0-100)
            
        Returns:
            Compressed JPEG bytes
        """
        quality = max(1, min(100, quality))
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', image, encode_params)
        return buffer.tobytes()
    
    def get_size_kb(self, data: bytes) -> float:
        """Get size in KB."""
        return len(data) / 1024.0
    
    def compress_to_target_size(self, image: np.ndarray, 
                                target_kb: float,
                                tolerance_percent: Optional[float] = None) -> CompressionResult:
        """
        Compress image to achieve target file size using binary search.
        
        Args:
            image: BGR numpy array
            target_kb: Target size in KB
            tolerance_percent: Acceptable deviation (default: class tolerance)
            
        Returns:
            CompressionResult with compressed data and metadata
        """
        if tolerance_percent is None:
            tolerance_percent = self.tolerance_percent
        
        # Calculate tolerance bounds
        min_size = target_kb * (1 - tolerance_percent / 100)
        max_size = target_kb * (1 + tolerance_percent / 100)
        
        # Binary search for quality
        low, high = 1, 100
        best_result = None
        best_diff = float('inf')
        iterations = 0
        max_iterations = 15
        
        while low <= high and iterations < max_iterations:
            iterations += 1
            mid = (low + high) // 2
            
            compressed = self.compress(image, mid)
            size_kb = self.get_size_kb(compressed)
            
            diff = abs(size_kb - target_kb)
            
            # Track best result
            if diff < best_diff:
                best_diff = diff
                best_result = CompressionResult(
                    image_bytes=compressed,
                    quality=mid,
                    file_size_kb=size_kb,
                    target_kb=target_kb,
                    within_tolerance=(min_size <= size_kb <= max_size),
                    iterations=iterations
                )
            
            # Check if within tolerance
            if min_size <= size_kb <= max_size:
                return best_result
            
            # Adjust search bounds
            if size_kb > target_kb:
                high = mid - 1
            else:
                low = mid + 1
        
        # Return best result found
        return best_result
    
    def compress_to_multiple_targets(self, image: np.ndarray,
                                      targets_kb: List[float]) -> Dict[float, CompressionResult]:
        """
        Compress image to multiple target sizes.
        
        Args:
            image: BGR numpy array
            targets_kb: List of target sizes in KB
            
        Returns:
            Dictionary mapping target size to CompressionResult
        """
        results = {}
        
        # Sort targets (largest first for efficiency)
        sorted_targets = sorted(targets_kb, reverse=True)
        
        for target in sorted_targets:
            result = self.compress_to_target_size(image, target)
            results[target] = result
        
        return results
    
    def estimate_quality_for_size(self, image: np.ndarray, 
                                   target_kb: float) -> int:
        """
        Estimate JPEG quality needed for target size.
        
        Args:
            image: BGR numpy array
            target_kb: Target size in KB
            
        Returns:
            Estimated quality value
        """
        # Quick sampling at a few quality levels
        samples = [(95, self.get_size_kb(self.compress(image, 95))),
                   (75, self.get_size_kb(self.compress(image, 75))),
                   (50, self.get_size_kb(self.compress(image, 50)))]
        
        # Linear interpolation/extrapolation
        for i in range(len(samples) - 1):
            q1, s1 = samples[i]
            q2, s2 = samples[i + 1]
            
            if s2 <= target_kb <= s1:
                # Interpolate
                ratio = (target_kb - s2) / (s1 - s2) if s1 != s2 else 0.5
                quality = int(q2 + ratio * (q1 - q2))
                return max(1, min(100, quality))
        
        # Extrapolate if outside range
        if target_kb > samples[0][1]:
            return 98
        else:
            # Use lowest sample as baseline
            ratio = target_kb / samples[-1][1]
            quality = int(samples[-1][0] * ratio)
            return max(1, min(100, quality))
    
    def get_quality_vs_size_curve(self, image: np.ndarray,
                                   quality_steps: int = 20) -> List[Tuple[int, float]]:
        """
        Generate quality vs file size data points.
        
        Args:
            image: BGR numpy array
            quality_steps: Number of quality levels to test
            
        Returns:
            List of (quality, size_kb) tuples
        """
        step = 100 // quality_steps
        results = []
        
        for q in range(step, 101, step):
            compressed = self.compress(image, q)
            size_kb = self.get_size_kb(compressed)
            results.append((q, size_kb))
        
        return results


def compress_to_target_size(image: np.ndarray, 
                           target_kb: float,
                           tolerance_percent: float = 5.0) -> Tuple[bytes, float, int]:
    """
    Convenience function to compress image to target size.
    
    Args:
        image: BGR numpy array
        target_kb: Target size in KB
        tolerance_percent: Acceptable deviation
        
    Returns:
        Tuple of (compressed_bytes, actual_size_kb, quality_used)
    """
    compressor = JPEGCompressor(tolerance_percent)
    result = compressor.compress_to_target_size(image, target_kb)
    return result.image_bytes, result.file_size_kb, result.quality

