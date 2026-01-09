# 📷 Digital Image Archiving System

## Complete Technical Documentation for Academic Report

A comprehensive Streamlit application for digitizing and enhancing old laminated/polaroid photographs. This system implements state-of-the-art computer vision algorithms for glare removal, perspective correction, color enhancement, intelligent compression, and AI-powered image analysis.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technical Implementation](#technical-implementation)
4. [Processing Pipeline](#processing-pipeline)
5. [Algorithms and Methods](#algorithms-and-methods)
6. [Quality Metrics](#quality-metrics)
7. [AI Integration](#ai-integration)
8. [Installation and Usage](#installation-and-usage)
9. [Configuration Options](#configuration-options)
10. [Project Structure](#project-structure)

---

## Project Overview

### Problem Statement

Old photographs, especially laminated or polaroid images, suffer from several degradation issues when digitized:
- **Glare and reflections** from scanning laminated surfaces
- **Perspective distortion** from non-perpendicular camera angles
- **Color fading** and color cast from aging
- **Noise and blur** from the digitization process
- **Storage constraints** requiring efficient compression

### Solution

This application provides an automated pipeline that:
1. Detects and removes glare from laminated surfaces
2. Corrects perspective distortion to produce rectangular outputs
3. Restores original colors and enhances faded photographs
4. Applies intelligent noise reduction and sharpening
5. Compresses images to target sizes while maintaining quality
6. (Optional) Uses AI to detect orientation and provide quality assessments

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Web Framework** | Streamlit | 1.28+ | Interactive web interface |
| **Image Processing** | OpenCV | 4.8+ | Core computer vision operations |
| **Numerical Computing** | NumPy | 1.24+ | Array operations and math |
| **Scientific Computing** | SciPy | 1.11+ | Advanced algorithms |
| **Image Analysis** | scikit-image | 0.21+ | Quality metrics (SSIM, PSNR) |
| **Image I/O** | Pillow | 10.0+ | Image format handling |
| **Visualization** | Plotly | 5.17+ | Interactive charts |
| **AI Analysis** | Anthropic API | Latest | Claude-powered image analysis |

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT WEB INTERFACE                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ File Upload │  │ Config Panel │  │ Results Display         │ │
│  └──────┬──────┘  └──────┬───────┘  └────────────▲────────────┘ │
└─────────┼────────────────┼───────────────────────┼──────────────┘
          │                │                       │
          ▼                ▼                       │
┌─────────────────────────────────────────────────┼──────────────┐
│                  AUTO-CONFIGURATION              │              │
│  ┌────────────────────────────────────────┐     │              │
│  │ Image Analysis → Parameter Optimization │     │              │
│  └────────────────────────────────────────┘     │              │
└─────────────────────────────────────────────────┼──────────────┘
                                                  │
┌─────────────────────────────────────────────────┼──────────────┐
│                 PROCESSING PIPELINE              │              │
│                                                  │              │
│  ┌──────────┐   ┌─────────────┐   ┌───────────┐ │              │
│  │   AI     │──▶│   Glare     │──▶│Perspective│─┼─┐            │
│  │Orientation│   │  Removal    │   │ Correction│ │ │            │
│  └──────────┘   └─────────────┘   └───────────┘ │ │            │
│                                                  │ │            │
│  ┌──────────┐   ┌─────────────┐   ┌───────────┐ │ │            │
│  │ Polaroid │◀──│ Enhancement │◀──│   Color   │◀┼─┘            │
│  │   Crop   │   │   (Smart)   │   │ Correction│ │              │
│  └────┬─────┘   └─────────────┘   └───────────┘ │              │
│       │                                          │              │
└───────┼──────────────────────────────────────────┼──────────────┘
        │                                          │
        ▼                                          │
┌─────────────────────────────────────────────────┼──────────────┐
│                    COMPRESSION                   │              │
│  ┌────────────────────────────────────────┐     │              │
│  │ JPEG Optimization with Quality Targets  │─────┘              │
│  │ • Binary search for optimal quality     │                    │
│  │ • PSNR/SSIM metric calculation          │                    │
│  └────────────────────────────────────────┘                    │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Image (JPEG/PNG/BMP/TIFF)
         │
         ▼
┌─────────────────────┐
│  Image Loading      │ → Convert to BGR (OpenCV format)
│  • Format detection │ → Store original for comparison
│  • Color space conv │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Auto-Configuration │ → Analyze: brightness, contrast, noise
│  • Brightness check │ → Detect: glare regions, color cast
│  • Noise estimation │ → Set: optimal processing parameters
│  • Glare detection  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Processing Steps   │ → Each step: before/after images
│  • Sequential exec  │ → Timing: millisecond precision
│  • Error handling   │ → Fallbacks: graceful degradation
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Compression        │ → Multiple target sizes
│  • Quality search   │ → Rate-distortion analysis
│  • Metric calc      │
└─────────────────────┘
         │
         ▼
Output: Enhanced images at various compression levels
```

---

## Technical Implementation

### Module Descriptions

#### 1. `config.py` - Auto-Configuration System

The auto-configuration module analyzes input images and determines optimal processing parameters.

**Key Functions:**

```python
class AutoConfig:
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Performs comprehensive image analysis.
        
        Returns:
        - brightness: Average pixel intensity (0-255)
        - contrast: Standard deviation of pixel values
        - noise_level: Estimated noise using Laplacian variance
        - glare_detected: Boolean indicating specular highlights
        - is_faded: Boolean indicating color degradation
        """
```

**Analysis Metrics:**

| Metric | Calculation | Threshold |
|--------|-------------|-----------|
| Brightness | `mean(grayscale)` | Low < 80, High > 180 |
| Contrast | `std(grayscale) / 127.5` | Low < 0.25 |
| Noise | `Laplacian variance / 1000` | High > 0.3 |
| Glare | Pixels > 250 count | > 1% of image |
| Faded | Saturation mean | < 0.3 indicates fading |

---

#### 2. `processors/glare_removal.py` - Glare Detection and Removal

**Algorithms Implemented:**

##### A. Adaptive Thresholding with Inpainting
```python
def remove_adaptive_inpaint(self, image: np.ndarray) -> np.ndarray:
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Detect glare using high threshold
    _, glare_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # 3. Dilate mask to cover surrounding affected pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)
    
    # 4. Apply inpainting using Navier-Stokes algorithm
    result = cv2.inpaint(image, glare_mask, inpaintRadius=5, 
                         flags=cv2.INPAINT_NS)
    return result
```

##### B. Specular Highlight Detection (LAB Color Space)
```python
def detect_specular_highlights(self, image: np.ndarray) -> np.ndarray:
    # 1. Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # 2. Specular highlights have high L and low chromatic variation
    high_l = l_channel > 240
    
    # 3. Low color saturation in specular regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_sat = hsv[:, :, 1] < 30
    
    # 4. Combine conditions
    specular_mask = (high_l & low_sat).astype(np.uint8) * 255
    return specular_mask
```

**Inpainting Methods:**

| Method | Algorithm | Best For |
|--------|-----------|----------|
| `cv2.INPAINT_NS` | Navier-Stokes based | Smooth regions |
| `cv2.INPAINT_TELEA` | Fast Marching Method | Textured regions |

---

#### 3. `processors/perspective.py` - Perspective Correction

**Detection Strategies:**

##### A. Contour-Based Detection
```python
def detect_contour(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
    # 1. Edge detection with auto-threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = np.median(gray)
    low = int(max(0, 0.67 * median))
    high = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, low, high)
    
    # 2. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Filter by area and approximate to polygon
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4:  # Quadrilateral found
            return self._order_points(approx), confidence
```

##### B. Hough Line Detection (Fallback)
```python
def detect_hough_lines(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
    # 1. Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=100,
                            minLineLength=min(h, w)//4,
                            maxLineGap=10)
    
    # 2. Classify lines as horizontal/vertical by angle
    # 3. Find intersections of border lines
    # 4. Return corner points
```

**Perspective Transform:**
```python
def apply_perspective_transform(self, image, corners):
    # Calculate destination size maintaining aspect ratio
    width, height = self._calculate_destination_size(corners)
    
    # Define destination points (rectangular)
    dst = np.array([[0, 0], [width-1, 0], 
                    [width-1, height-1], [0, height-1]], dtype=np.float32)
    
    # Compute transformation matrix
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # Apply warp
    result = cv2.warpPerspective(image, M, (width, height))
    return result
```

---

#### 4. `processors/color_correction.py` - Color Restoration

**White Balance Algorithms:**

##### A. Gray World Assumption
```python
def gray_world(self, image: np.ndarray) -> np.ndarray:
    """
    Assumes average color should be neutral gray.
    Adjusts each channel to have equal means.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    
    # Calculate channel means
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg = (avg_b + avg_g + avg_r) / 3
    
    # Scale channels to have equal means
    b = np.clip(b * (avg / avg_b), 0, 255)
    g = np.clip(g * (avg / avg_g), 0, 255)
    r = np.clip(r * (avg / avg_r), 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)
```

##### B. White Patch Algorithm
```python
def white_patch(self, image: np.ndarray) -> np.ndarray:
    """
    Assumes brightest point should be white.
    Scales channels based on maximum values.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    
    # Use 99th percentile to avoid outliers
    max_b = np.percentile(b, 99)
    max_g = np.percentile(g, 99)
    max_r = np.percentile(r, 99)
    
    # Scale to make white point (255, 255, 255)
    b = np.clip(b * (255 / max_b), 0, 255)
    g = np.clip(g * (255 / max_g), 0, 255)
    r = np.clip(r * (255 / max_r), 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)
```

##### C. CLAHE (Contrast Limited Adaptive Histogram Equalization)
```python
def apply_clahe(self, image: np.ndarray) -> np.ndarray:
    """
    Enhances local contrast without over-amplifying noise.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```

**Color Cast Removal:**
```python
def remove_color_cast(self, image: np.ndarray) -> np.ndarray:
    """
    Detects and removes color cast using LAB color space.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Center A and B channels around 128 (neutral)
    a_mean, b_mean = np.mean(a), np.mean(b)
    a = np.clip(a - (a_mean - 128), 0, 255).astype(np.uint8)
    b = np.clip(b - (b_mean - 128), 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```

---

#### 5. `processors/enhancement.py` - AI-Powered Image Enhancement

**Simple Enhancement with AI Recommendations:**

Since the color correction step already handles most improvements (white balance, CLAHE, color cast removal), the enhancement module provides optional light touch-ups based on **AI analysis** using Claude.

```python
@dataclass 
class EnhancementResult:
    image: np.ndarray
    denoise_applied: str      # "none", "light", "medium"
    sharpen_applied: str      # "none", "light", "medium"
    ai_recommendations: str   # AI explanation
    processing_time: float
```

**AI-Powered Recommendations:**

The module sends the image to Claude for analysis:

```python
def get_ai_recommendations(self, image: np.ndarray, api_key: str) -> Dict:
    """
    Use Claude AI to analyze image and recommend enhancement settings.
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {...}},
                {"type": "text", "text": """
                    Analyze this photograph and recommend enhancement settings.
                    The image has already been color corrected.
                    Focus ONLY on:
                    1. Noise level - is denoising needed?
                    2. Sharpness - is sharpening needed?
                    
                    Be conservative - only recommend if clearly needed.
                    Respond in JSON: {"denoise": "none|light|medium", 
                                      "sharpen": "none|light|medium", 
                                      "reason": "explanation"}
                """}
            ]
        }]
    )
    return json.loads(response.content[0].text)
```

**Simple Processing Methods:**

| Method | Technique | Description |
|--------|-----------|-------------|
| `denoise_light` | `fastNlMeansDenoisingColored(h=3)` | Gentle noise reduction |
| `denoise_medium` | Bilateral + NL-means | Edge-preserving smoothing |
| `sharpen_light` | Unsharp mask (0.3) | Subtle edge enhancement |
| `sharpen_medium` | Unsharp mask (0.5) | Moderate sharpening |

**Fallback Auto Mode:**

If AI is unavailable, uses simple heuristics:
```python
# Auto denoising based on Laplacian variance
noise = self.estimate_noise_level(image)
if noise > 0.4:
    result = self.denoise_medium(result)
elif noise > 0.2:
    result = self.denoise_light(result)

# Auto sharpening based on sharpness measurement
sharpness = self.measure_sharpness(result)
if sharpness < 0.3:
    result = self.sharpen_light(result)
```

---

#### 6. `processors/perspective.py` - Polaroid Content Cropper

**Multi-Strategy Detection System:**

The polaroid cropper uses five different detection strategies for robust content extraction:

##### Strategy 1: Color Segmentation
```python
def _detect_by_color_segmentation(self, image, gray):
    """
    Segments content from border based on color uniformity.
    Borders (especially white) have low color variance.
    """
    # Convert to LAB for perceptual color distance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Sample border color (average of all edges)
    border_color = self._sample_border_color(lab)
    
    # Calculate per-pixel distance from border color
    diff = np.sqrt(np.sum((lab - border_color) ** 2, axis=2))
    
    # Threshold to separate content
    content_mask = (diff > threshold).astype(np.uint8) * 255
    
    # Find largest centered contour in mask
    return self._find_content_region(content_mask)
```

##### Strategy 2: Rectangular Contour Detection
```python
def _detect_by_rectangular_contour(self, gray):
    """
    Looks for rectangular shapes inside the image borders.
    """
    # Multi-scale edge detection
    for canny_low, canny_high in [(30, 80), (50, 150), (80, 200)]:
        edges = cv2.Canny(gray, canny_low, canny_high)
        contours = cv2.findContours(edges, cv2.RETR_LIST, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for 4-6 vertex polygons (approximately rectangular)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if 4 <= len(approx) <= 6:
                # Validate: inside borders, reasonable aspect ratio
                if self._validate_rectangle(approx, gray.shape):
                    return cv2.boundingRect(approx)
```

##### Strategy 3: Gradient Transition Detection
```python
def _detect_by_gradient_transition(self, gray):
    """
    Finds content boundaries by locating strong gradient transitions.
    """
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Project onto rows/columns to find boundaries
    row_gradient = np.mean(grad_mag, axis=1)
    col_gradient = np.mean(grad_mag, axis=0)
    
    # Find peaks indicating border-content transitions
    return self._find_transition_points(row_gradient, col_gradient)
```

##### Strategy 4: Edge Density Analysis
```python
def _detect_by_edge_density(self, gray):
    """
    Content regions have higher edge density than uniform borders.
    """
    edges = cv2.Canny(gray, 30, 100)
    
    # Calculate cumulative edge density from borders
    col_density = np.sum(edges, axis=0) / gray.shape[0]
    row_density = np.sum(edges, axis=1) / gray.shape[1]
    
    # Find where density increases (content begins)
    return self._find_density_boundaries(row_density, col_density)
```

**Detection Scoring:**
```python
def detect_polaroid_content(self, image):
    results = []
    
    for method_func, method_name in detection_methods:
        result = method_func(image)
        if result:
            # Score based on centrality and size
            score = self._calculate_detection_score(result, image.shape)
            results.append((result, method_name, score))
    
    # Return best scoring result
    return max(results, key=lambda x: x[2])
```

---

#### 7. `compression/` - Intelligent Compression

**Binary Search for Target Size:**

```python
def compress_to_target_size(image: np.ndarray, target_kb: int) -> Tuple[bytes, int]:
    """
    Uses binary search to find optimal JPEG quality for target file size.
    """
    target_bytes = target_kb * 1024
    
    quality_low, quality_high = 1, 100
    best_result = None
    
    while quality_low <= quality_high:
        quality = (quality_low + quality_high) // 2
        
        # Encode at current quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', image, encode_params)
        
        current_size = len(buffer)
        
        if current_size <= target_bytes:
            best_result = (bytes(buffer), quality)
            quality_low = quality + 1  # Try higher quality
        else:
            quality_high = quality - 1  # Need lower quality
    
    return best_result
```

---

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)

```python
def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    PSNR measures the ratio between maximum signal power and noise power.
    
    Formula: PSNR = 10 * log10(MAX² / MSE)
    
    Where:
    - MAX = 255 for 8-bit images
    - MSE = Mean Squared Error between images
    
    Interpretation:
    - > 40 dB: Excellent quality (imperceptible loss)
    - 30-40 dB: Good quality (minor artifacts)
    - 20-30 dB: Acceptable quality (visible artifacts)
    - < 20 dB: Poor quality (significant degradation)
    """
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))
```

### SSIM (Structural Similarity Index)

```python
from skimage.metrics import structural_similarity

def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    SSIM measures perceptual similarity considering:
    - Luminance (brightness patterns)
    - Contrast (local variance)
    - Structure (correlation patterns)
    
    Formula: SSIM = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
    
    Interpretation:
    - 1.0: Identical images
    - > 0.95: Excellent quality
    - 0.90-0.95: Good quality
    - 0.80-0.90: Acceptable quality
    - < 0.80: Poor quality
    """
    return structural_similarity(original, compressed, 
                                 channel_axis=2,  # Color images
                                 data_range=255)
```

### Rate-Distortion Curve

The application generates interactive rate-distortion curves showing the trade-off between file size (rate) and quality (distortion):

```python
def plot_rate_distortion_curve(compression_results: List[Dict]) -> Figure:
    """
    Creates interactive Plotly chart with:
    - X-axis: File size (KB)
    - Y-axis: PSNR (dB) or SSIM
    - Hover info: Quality setting, exact metrics
    """
    sizes = [r['size_kb'] for r in compression_results]
    psnr_values = [r['psnr'] for r in compression_results]
    ssim_values = [r['ssim'] for r in compression_results]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=sizes, y=psnr_values, name="PSNR"))
    fig.add_trace(go.Scatter(x=sizes, y=ssim_values, name="SSIM"), 
                  secondary_y=True)
    
    return fig
```

---

## AI Integration

### Anthropic Claude Integration

The application uses Claude's vision capabilities for:

#### 1. Orientation Detection
```python
def ai_detect_orientation(self, image_bytes: bytes, api_key: str) -> Dict:
    """
    Sends image to Claude for orientation analysis.
    
    Prompt:
    "Analyze the orientation of this photograph.
    Is it correctly oriented (0 degrees rotation needed), 
    or does it need to be rotated 90, 180, or 270 degrees clockwise?
    Focus on human subjects, text, and natural scene elements."
    
    Returns: {"rotation_angle": N, "reason": "..."}
    Where N is one of [0, 90, 180, 270]
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {...}},
                {"type": "text", "text": orientation_prompt}
            ]
        }]
    )
    
    return json.loads(response.content[0].text)
```

#### 2. Quality Assessment
```python
def ai_analyze_image(self, image_bytes: bytes, api_key: str) -> Dict:
    """
    Comprehensive AI analysis providing:
    - quality_score: 1-10 rating
    - issues: List of detected problems
    - recommendations: Suggested improvements
    - description: Content description
    """
```

---

## Installation and Usage

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation Steps

```bash
# 1. Clone or navigate to project
cd photo_archiver

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py --server.port 8502
```

### Dependencies (`requirements.txt`)

```
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
scikit-image>=0.21.0
Pillow>=10.0.0
plotly>=5.17.0
anthropic>=0.5.0
```

---

## Configuration Options

### Sidebar Controls

| Option | Type | Values | Description |
|--------|------|--------|-------------|
| AI Orientation | Toggle | On/Off | Use AI to detect and fix rotation |
| Glare Removal | Select | Auto/Adaptive/Inpainting | Glare removal method |
| Inpaint Radius | Slider | 1-15 | Size of inpainting neighborhood |
| White Balance | Select | Auto/Gray World/White Patch | Color correction method |
| Saturation | Slider | 0.8-1.5 | Color saturation adjustment |
| Denoise | Select | Auto/Off/Light/Medium/Strong | Noise reduction strength |
| Sharpen | Select | Auto/Off/Light/Medium/Strong | Sharpening intensity |
| Polaroid Crop | Toggle | On/Off | Extract content from frame |

### Target Compression Sizes

| Target | Use Case | Typical Quality |
|--------|----------|-----------------|
| 30 KB | Thumbnails, web previews | PSNR ~28-32 dB |
| 100 KB | Web sharing, email | PSNR ~32-36 dB |
| 500 KB | High-quality web | PSNR ~36-40 dB |
| 1000 KB | Near-archival | PSNR ~40+ dB |

---

## Project Structure

```
photo_archiver/
├── app.py                          # Main Streamlit application (500+ lines)
│   ├── Page configuration & styling
│   ├── Sidebar controls
│   ├── File upload handling
│   ├── process_image() pipeline
│   ├── display_pipeline_results()
│   └── Compression analysis UI
│
├── config.py                       # Auto-configuration system
│   ├── DEFAULT_CONFIG dictionary
│   └── AutoConfig class
│
├── requirements.txt                # Python dependencies
│
├── processors/                     # Image processing modules
│   ├── __init__.py                # Module exports
│   │
│   ├── glare_removal.py           # Glare detection and removal
│   │   ├── GlareRemover class
│   │   ├── detect_glare_regions()
│   │   ├── remove_adaptive_inpaint()
│   │   └── detect_specular_highlights()
│   │
│   ├── perspective.py             # Perspective correction & polaroid crop
│   │   ├── PerspectiveCorrector class
│   │   │   ├── detect_contour()
│   │   │   ├── detect_hough_lines()
│   │   │   └── apply_perspective_transform()
│   │   └── PolaroidCropper class
│   │       ├── _detect_by_color_segmentation()
│   │       ├── _detect_by_rectangular_contour()
│   │       ├── _detect_by_gradient_transition()
│   │       └── _detect_by_edge_density()
│   │
│   ├── color_correction.py        # Color restoration
│   │   ├── ColorCorrector class
│   │   ├── gray_world()
│   │   ├── white_patch()
│   │   ├── apply_clahe()
│   │   └── remove_color_cast()
│   │
│   ├── enhancement.py             # Smart image enhancement
│   │   ├── ImageEnhancer class
│   │   ├── ImageCharacteristics dataclass
│   │   ├── analyze_image_characteristics()
│   │   ├── get_smart_enhancement_params()
│   │   ├── denoise_light/medium/strong()
│   │   └── sharpen_unsharp_mask/edge_aware()
│   │
│   └── ai_enhance.py              # AI-powered analysis
│       ├── AIEnhancer class
│       ├── ai_detect_orientation()
│       ├── ai_analyze_image()
│       └── rotate_image()
│
├── compression/                    # Compression modules
│   ├── __init__.py
│   │
│   ├── jpeg_compressor.py         # JPEG optimization
│   │   └── compress_to_target_size()
│   │
│   └── metrics.py                 # Quality metrics
│       ├── calculate_psnr()
│       ├── calculate_ssim()
│       └── calculate_metrics()
│
└── utils/                          # Utility modules
    ├── __init__.py
    │
    ├── image_utils.py             # Image I/O helpers
    │   ├── load_image()
    │   ├── image_to_bytes()
    │   ├── bgr_to_pil()
    │   └── get_image_info()
    │
    └── visualization.py           # Display utilities
        ├── plot_rate_distortion_curve()
        ├── create_comparison_slider()
        ├── create_side_by_side()
        └── create_metrics_dashboard()
```

---

## Acknowledgments

- **OpenCV**: Open Source Computer Vision Library for core image processing
- **Streamlit**: Open-source app framework for data science
- **Anthropic**: Claude AI for intelligent image analysis
- **scikit-image**: Image processing algorithms and quality metrics
- **Plotly**: Interactive visualization library

---

## References

1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer.
3. Wang, Z., et al. (2004). "Image quality assessment: from error visibility to structural similarity." *IEEE Transactions on Image Processing*.
4. Telea, A. (2004). "An Image Inpainting Technique Based on the Fast Marching Method." *Journal of Graphics Tools*.
5. Bertalmio, M., et al. (2001). "Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting." *IEEE CVPR*.

---

*This documentation was prepared for academic reporting purposes. The system demonstrates practical application of computer vision techniques for digital preservation of photographic materials.*
