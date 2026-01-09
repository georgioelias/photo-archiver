"""
Common image utility functions for the photo archiver.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
import io
from PIL import Image


def load_image(source: Union[str, bytes, io.BytesIO, np.ndarray]) -> np.ndarray:
    """
    Load image from various sources.
    
    Args:
        source: File path, bytes, BytesIO stream, or numpy array
        
    Returns:
        BGR numpy array
    """
    if isinstance(source, np.ndarray):
        return source
    
    if isinstance(source, str):
        # File path
        image = cv2.imread(source)
        if image is None:
            raise ValueError(f"Could not load image from: {source}")
        return image
    
    if isinstance(source, bytes):
        # Convert bytes to numpy array
        nparr = np.frombuffer(source, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes")
        return image
    
    if isinstance(source, io.BytesIO):
        # BytesIO stream
        source.seek(0)
        data = source.read()
        return load_image(data)
    
    # Try to read from file-like object
    if hasattr(source, 'read'):
        data = source.read()
        return load_image(data)
    
    raise ValueError(f"Unsupported image source type: {type(source)}")


def save_image(image: np.ndarray, path: str, quality: int = 95) -> bool:
    """
    Save image to file.
    
    Args:
        image: BGR numpy array
        path: Output file path
        quality: JPEG quality (0-100)
        
    Returns:
        True if successful
    """
    # Determine format from extension
    ext = path.lower().split('.')[-1]
    
    if ext in ['jpg', 'jpeg']:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == 'png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - int(quality / 12)]
    else:
        params = []
    
    return cv2.imwrite(path, image, params)


def resize_image(image: np.ndarray, 
                 max_dimension: Optional[int] = None,
                 scale: Optional[float] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Resize image with various options.
    
    Args:
        image: Input image
        max_dimension: Maximum width or height
        scale: Scale factor
        width: Target width (maintains aspect ratio if height not set)
        height: Target height (maintains aspect ratio if width not set)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if scale is not None:
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif max_dimension is not None:
        if max(h, w) <= max_dimension:
            return image
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif width is not None and height is not None:
        new_w = width
        new_h = height
    elif width is not None:
        scale = width / w
        new_w = width
        new_h = int(h * scale)
    elif height is not None:
        scale = height / h
        new_h = height
        new_w = int(w * scale)
    else:
        return image
    
    # Choose interpolation based on scaling direction
    if new_w < w or new_h < h:
        interp = cv2.INTER_AREA  # Best for downscaling
    else:
        interp = cv2.INTER_LANCZOS4  # Best for upscaling
    
    if interpolation != cv2.INTER_AREA:
        interp = interpolation
    
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def convert_color_space(image: np.ndarray, 
                        from_space: str = "BGR",
                        to_space: str = "RGB") -> np.ndarray:
    """
    Convert between color spaces.
    
    Args:
        image: Input image
        from_space: Source color space (BGR, RGB, HSV, LAB, GRAY)
        to_space: Target color space
        
    Returns:
        Converted image
    """
    conversion_map = {
        ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
        ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
        ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
        ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
        ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
        ("HSV", "BGR"): cv2.COLOR_HSV2BGR,
        ("BGR", "LAB"): cv2.COLOR_BGR2LAB,
        ("LAB", "BGR"): cv2.COLOR_LAB2BGR,
        ("RGB", "HSV"): cv2.COLOR_RGB2HSV,
        ("HSV", "RGB"): cv2.COLOR_HSV2RGB,
        ("RGB", "LAB"): cv2.COLOR_RGB2LAB,
        ("LAB", "RGB"): cv2.COLOR_LAB2RGB,
    }
    
    key = (from_space.upper(), to_space.upper())
    
    if key not in conversion_map:
        raise ValueError(f"Unsupported color space conversion: {from_space} -> {to_space}")
    
    return cv2.cvtColor(image, conversion_map[key])


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is uint8 type with values in [0, 255].
    
    Args:
        image: Input image
        
    Returns:
        uint8 image
    """
    if image.dtype == np.uint8:
        return image
    
    # Handle float images (assume 0-1 range)
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255)
    
    return image.astype(np.uint8)


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate image for processing.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, np.ndarray):
        return False, f"Image is not a numpy array: {type(image)}"
    
    if len(image.shape) < 2:
        return False, f"Image has invalid dimensions: {image.shape}"
    
    if image.size == 0:
        return False, "Image is empty"
    
    # Check for NaN or Inf values
    if np.isnan(image).any():
        return False, "Image contains NaN values"
    
    if np.isinf(image).any():
        return False, "Image contains Inf values"
    
    # Check dimensions
    h, w = image.shape[:2]
    if h < 10 or w < 10:
        return False, f"Image too small: {w}x{h}"
    
    if h > 20000 or w > 20000:
        return False, f"Image too large: {w}x{h}"
    
    return True, "OK"


def image_to_bytes(image: np.ndarray, 
                   format: str = "JPEG",
                   quality: int = 95) -> bytes:
    """
    Convert image to bytes.
    
    Args:
        image: BGR numpy array
        format: Output format (JPEG, PNG)
        quality: Compression quality
        
    Returns:
        Image bytes
    """
    if format.upper() == "JPEG":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = ".jpg"
    elif format.upper() == "PNG":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - int(quality / 12)]
        ext = ".png"
    else:
        params = []
        ext = f".{format.lower()}"
    
    success, buffer = cv2.imencode(ext, image, params)
    if not success:
        raise ValueError(f"Failed to encode image as {format}")
    
    return buffer.tobytes()


def bytes_to_image(data: bytes) -> np.ndarray:
    """
    Convert bytes to image.
    
    Args:
        data: Image bytes
        
    Returns:
        BGR numpy array
    """
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image from bytes")
    
    return image


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert BGR numpy array to PIL Image.
    
    Args:
        image: BGR numpy array
        
    Returns:
        PIL Image (RGB)
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to BGR numpy array.
    
    Args:
        image: PIL Image
        
    Returns:
        BGR numpy array
    """
    rgb = np.array(image)
    if len(rgb.shape) == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    # Calculate statistics
    if channels == 3:
        means = np.mean(image, axis=(0, 1))
        stds = np.std(image, axis=(0, 1))
    else:
        means = [np.mean(image)]
        stds = [np.std(image)]
    
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(image.dtype),
        "size_bytes": image.nbytes,
        "aspect_ratio": round(w / h, 3),
        "mean_brightness": round(float(np.mean(means)), 2),
        "std_brightness": round(float(np.mean(stds)), 2),
        "min_value": int(image.min()),
        "max_value": int(image.max()),
    }


def create_thumbnail(image: np.ndarray, 
                     size: Tuple[int, int] = (200, 200),
                     maintain_aspect: bool = True) -> np.ndarray:
    """
    Create a thumbnail of the image.
    
    Args:
        image: Input image
        size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Thumbnail image
    """
    target_w, target_h = size
    h, w = image.shape[:2]
    
    if maintain_aspect:
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center thumbnail
        if len(image.shape) == 3:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail
        return canvas
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

