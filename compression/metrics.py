"""
Image quality metrics for compression analysis.
Implements PSNR, SSIM, MSE, and other quality measures.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Complete quality metrics for an image comparison."""
    psnr: float
    ssim: float
    mse: float
    file_size_kb: float
    compression_ratio: float
    bits_per_pixel: float


def calculate_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        original: Original image
        compressed: Compressed/processed image
        
    Returns:
        MSE value (lower is better, 0 = identical)
    """
    # Ensure same dimensions
    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    # Calculate MSE
    diff = original.astype(np.float64) - compressed.astype(np.float64)
    mse = np.mean(diff ** 2)
    
    return float(mse)


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original: Original image
        compressed: Compressed/processed image
        
    Returns:
        PSNR value in dB (higher is better, inf = identical)
    """
    mse = calculate_mse(original, compressed)
    
    if mse == 0:
        return float('inf')
    
    # Max pixel value
    max_pixel = 255.0
    
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    
    return float(psnr)


def calculate_ssim(original: np.ndarray, compressed: np.ndarray,
                   window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        original: Original image
        compressed: Compressed/processed image
        window_size: Size of the sliding window
        
    Returns:
        SSIM value (0-1, higher is better, 1 = identical)
    """
    # Ensure same dimensions
    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale if color
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        compressed_gray = compressed
    
    # Convert to float
    img1 = original_gray.astype(np.float64)
    img2 = compressed_gray.astype(np.float64)
    
    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Create Gaussian kernel
    sigma = 1.5
    kernel_size = window_size
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    window = np.outer(kernel, kernel.transpose())
    
    # Calculate means
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    ssim = np.mean(ssim_map)
    
    return float(ssim)


def calculate_ssim_skimage(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate SSIM using scikit-image (more accurate but slower).
    
    Args:
        original: Original image
        compressed: Compressed/processed image
        
    Returns:
        SSIM value (0-1)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Ensure same dimensions
        if original.shape != compressed.shape:
            compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
        
        # Handle color images
        if len(original.shape) == 3:
            return float(ssim(original, compressed, channel_axis=2, data_range=255))
        else:
            return float(ssim(original, compressed, data_range=255))
            
    except ImportError:
        # Fall back to custom implementation
        return calculate_ssim(original, compressed)


def calculate_compression_ratio(original_size_bytes: int, 
                                compressed_size_bytes: int) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original_size_bytes: Size of original image
        compressed_size_bytes: Size of compressed image
        
    Returns:
        Compression ratio (original/compressed)
    """
    if compressed_size_bytes == 0:
        return float('inf')
    
    return original_size_bytes / compressed_size_bytes


def calculate_bits_per_pixel(compressed_size_bytes: int,
                             width: int, height: int) -> float:
    """
    Calculate bits per pixel.
    
    Args:
        compressed_size_bytes: Size of compressed image
        width: Image width
        height: Image height
        
    Returns:
        Bits per pixel
    """
    total_pixels = width * height
    if total_pixels == 0:
        return 0.0
    
    total_bits = compressed_size_bytes * 8
    return total_bits / total_pixels


def calculate_metrics(original: np.ndarray, 
                     compressed: np.ndarray,
                     compressed_size_bytes: Optional[int] = None,
                     use_skimage_ssim: bool = True) -> QualityMetrics:
    """
    Calculate all quality metrics between original and compressed images.
    
    Args:
        original: Original image (BGR numpy array)
        compressed: Compressed image (BGR numpy array)
        compressed_size_bytes: Size of compressed file (optional)
        use_skimage_ssim: Use scikit-image for SSIM (more accurate)
        
    Returns:
        QualityMetrics with all calculated values
    """
    # Calculate basic metrics
    psnr = calculate_psnr(original, compressed)
    mse = calculate_mse(original, compressed)
    
    # Calculate SSIM
    if use_skimage_ssim:
        ssim = calculate_ssim_skimage(original, compressed)
    else:
        ssim = calculate_ssim(original, compressed)
    
    # Calculate size-related metrics if compressed size provided
    if compressed_size_bytes is not None:
        file_size_kb = compressed_size_bytes / 1024.0
        
        # Original size (uncompressed BGR)
        original_size_bytes = original.size  # total bytes in array
        compression_ratio = calculate_compression_ratio(original_size_bytes, compressed_size_bytes)
        
        # Bits per pixel
        height, width = original.shape[:2]
        bits_per_pixel = calculate_bits_per_pixel(compressed_size_bytes, width, height)
    else:
        file_size_kb = 0.0
        compression_ratio = 0.0
        bits_per_pixel = 0.0
    
    return QualityMetrics(
        psnr=psnr,
        ssim=ssim,
        mse=mse,
        file_size_kb=file_size_kb,
        compression_ratio=compression_ratio,
        bits_per_pixel=bits_per_pixel
    )


def get_quality_assessment(metrics: QualityMetrics) -> Dict[str, Any]:
    """
    Get human-readable quality assessment based on metrics.
    
    Args:
        metrics: Calculated quality metrics
        
    Returns:
        Dictionary with assessment details
    """
    # PSNR assessment (typical ranges)
    # >40 dB: Excellent (imperceptible difference)
    # 35-40 dB: Very good
    # 30-35 dB: Good
    # 25-30 dB: Fair
    # <25 dB: Poor
    
    if metrics.psnr == float('inf'):
        psnr_rating = "Perfect (identical)"
    elif metrics.psnr > 40:
        psnr_rating = "Excellent"
    elif metrics.psnr > 35:
        psnr_rating = "Very Good"
    elif metrics.psnr > 30:
        psnr_rating = "Good"
    elif metrics.psnr > 25:
        psnr_rating = "Fair"
    else:
        psnr_rating = "Poor"
    
    # SSIM assessment
    # >0.95: Excellent
    # 0.9-0.95: Very Good
    # 0.85-0.9: Good
    # 0.8-0.85: Fair
    # <0.8: Poor
    
    if metrics.ssim > 0.99:
        ssim_rating = "Excellent (near identical)"
    elif metrics.ssim > 0.95:
        ssim_rating = "Excellent"
    elif metrics.ssim > 0.90:
        ssim_rating = "Very Good"
    elif metrics.ssim > 0.85:
        ssim_rating = "Good"
    elif metrics.ssim > 0.80:
        ssim_rating = "Fair"
    else:
        ssim_rating = "Poor"
    
    # Overall recommendation
    overall_score = (metrics.psnr / 50 * 0.4 + metrics.ssim * 0.6) * 10
    overall_score = min(10, max(0, overall_score))
    
    return {
        "psnr_rating": psnr_rating,
        "ssim_rating": ssim_rating,
        "overall_score": round(overall_score, 1),
        "recommendation": "Acceptable for archival" if overall_score > 6 else "Consider higher quality",
        "details": {
            "psnr_db": round(metrics.psnr, 2) if metrics.psnr != float('inf') else "Identical",
            "ssim_index": round(metrics.ssim, 4),
            "mse": round(metrics.mse, 2),
            "size_kb": round(metrics.file_size_kb, 1),
            "compression_ratio": round(metrics.compression_ratio, 1),
            "bits_per_pixel": round(metrics.bits_per_pixel, 2)
        }
    }


def find_optimal_quality_point(quality_metrics_list: list) -> Dict[str, Any]:
    """
    Find the optimal quality/size trade-off point.
    
    Args:
        quality_metrics_list: List of (file_size_kb, QualityMetrics) tuples
        
    Returns:
        Dictionary with optimal point recommendation
    """
    if not quality_metrics_list:
        return {"error": "No data provided"}
    
    # Calculate efficiency scores
    scored_points = []
    
    for size_kb, metrics in quality_metrics_list:
        # Efficiency = quality gained per KB
        # We want high SSIM and PSNR with low file size
        
        # Normalize PSNR (assume max useful is ~50 dB)
        psnr_norm = min(metrics.psnr / 50, 1.0) if metrics.psnr != float('inf') else 1.0
        
        # SSIM is already 0-1
        ssim_norm = metrics.ssim
        
        # Quality score (weighted average)
        quality_score = psnr_norm * 0.4 + ssim_norm * 0.6
        
        # Size penalty (logarithmic - diminishing returns for larger files)
        size_penalty = np.log10(size_kb + 1) / np.log10(1001)  # Normalize to ~0-1 for 1KB-1MB
        
        # Efficiency score
        efficiency = quality_score / (size_penalty + 0.1)  # +0.1 to avoid division issues
        
        scored_points.append({
            "size_kb": size_kb,
            "metrics": metrics,
            "quality_score": quality_score,
            "efficiency": efficiency
        })
    
    # Sort by efficiency
    scored_points.sort(key=lambda x: x["efficiency"], reverse=True)
    
    # Find "knee" point where quality starts to plateau
    # This is typically the optimal balance point
    optimal = scored_points[0]
    
    # Also find minimum acceptable quality point
    min_acceptable = None
    for point in reversed(scored_points):
        if point["metrics"].ssim > 0.85 and point["metrics"].psnr > 30:
            min_acceptable = point
            break
    
    return {
        "optimal": {
            "size_kb": optimal["size_kb"],
            "psnr": optimal["metrics"].psnr,
            "ssim": optimal["metrics"].ssim,
            "efficiency": optimal["efficiency"]
        },
        "minimum_acceptable": {
            "size_kb": min_acceptable["size_kb"] if min_acceptable else None,
            "psnr": min_acceptable["metrics"].psnr if min_acceptable else None,
            "ssim": min_acceptable["metrics"].ssim if min_acceptable else None
        } if min_acceptable else None,
        "all_points": [
            {
                "size_kb": p["size_kb"],
                "quality_score": round(p["quality_score"], 3),
                "efficiency": round(p["efficiency"], 3)
            }
            for p in scored_points
        ]
    }

