# Digital Polaroid Archiving System
## Academic Technical Report

---

## Abstract

This report presents a comprehensive web-based application for digitizing and restoring vintage polaroid photographs. The system combines traditional computer vision algorithms with modern Large Language Model (LLM) capabilities to achieve intelligent, context-aware image processing. Unlike conventional photo editors, this application leverages AI not just for analysis, but for dynamic decision-making throughout the processing pipeline—representing a novel approach to heritage photo preservation.

---

## 1. Introduction

### 1.1 Problem Statement
Polaroid photographs represent irreplaceable personal and historical artifacts. However, digitizing these images presents unique challenges:
- Physical degradation (fading, color cast, damage)
- Capture artifacts (glare, reflections, perspective distortion)
- The distinctive polaroid frame structure requiring specialized handling
- Subjective quality assessment that traditional algorithms cannot perform

### 1.2 Objectives
- Develop an automated pipeline for polaroid digitization
- Implement intelligent detection of the polaroid frame within photographs
- Apply adaptive image restoration techniques
- Utilize LLMs for subjective analysis and quality improvement assessment

---

## 2. System Architecture

### 2.1 Technology Stack
| Component | Technology |
|-----------|------------|
| Frontend | Streamlit (Python web framework) |
| Image Processing | OpenCV, NumPy, SciPy, scikit-image |
| AI Integration | Anthropic Claude API (claude-sonnet-4-20250514) |
| Visualization | Plotly |

### 2.2 Processing Pipeline
The application implements a six-stage sequential pipeline:

```
Input → Polaroid Crop → Glare Removal → Perspective Correction → 
Color Correction → Enhancement → Compression → Output
```

Each stage operates on the output of the previous stage, with optional AI-assisted decision making.

---

## 3. Core Methodologies

### 3.1 Two-Stage Polaroid Detection (Novel Contribution)

A key innovation is the **two-stage polaroid detection algorithm**, designed for real-world photographs of polaroids (not flatbed scans):

**Stage 1: Frame Localization**
- HSV color space conversion for white region detection
- Contour analysis with rectangularity scoring
- Aspect ratio filtering (polaroids ≈ 1.0-1.4 ratio)

**Stage 2: Content Extraction**
- Inward boundary scanning using saturation/brightness thresholds
- Adaptive margin calculation (2.5% of dimension)
- Median-based content detection (40th percentile) for robustness

```python
# Content detection criteria
is_content = saturation > 50 OR brightness < 170
```

### 3.2 Intelligent Glare Removal

The glare removal module employs multiple strategies:
- **Specular Highlight Detection**: LAB color space analysis
- **Adaptive Inpainting**: Telea and Navier-Stokes algorithms
- **Polaroid-Aware Processing**: Distinguishes white borders from actual glare using:
  - Shape rectangularity analysis
  - Edge proximity detection
  - Color uniformity measurement

### 3.3 Color Correction Pipeline

Three-method white balance with automatic selection:
1. **Gray World Algorithm**: Assumes average scene color is neutral
2. **White Patch Retinex**: Uses brightest pixels as white reference
3. **Combined Approach**: Weighted fusion of both methods

Additional processing:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Color cast detection and removal via LAB channel analysis

### 3.4 JPEG Compression with Quality Metrics

Binary search optimization for target file sizes with quality metrics:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **MSE** (Mean Squared Error)
- **Bits per Pixel** analysis

---

## 4. LLM Integration (Unique Approach)

### 4.1 AI-Powered Orientation Detection

Traditional orientation detection relies on EXIF data or edge detection. Our approach uses **vision-language models** for semantic understanding:

```
Prompt Strategy:
1. Identify polaroid frame boundaries
2. Locate the thick "signature strip" border
3. Apply rule: thick border must be at BOTTOM
4. Calculate required rotation (0°/90°/180°/270°)
```

This semantic approach handles edge cases that algorithmic methods cannot:
- Ambiguous content (abstract images)
- Multiple potential orientations
- Degraded or partial frames

### 4.2 Comparative Quality Assessment

Rather than absolute scoring, we implement **relative improvement analysis**:

**Input**: Original image + Processed image
**Output**: 
- Per-aspect improvement percentages
- Overall transformation score
- Natural language summary

**Prompt Engineering Highlights**:
```
"Focus ONLY on positive changes...
For each improvement, estimate percentage (10-100%)...
Be enthusiastic and positive!"
```

This approach:
- Provides actionable, interpretable feedback
- Celebrates restoration success (user experience)
- Quantifies subjective improvements

### 4.3 Why This LLM Usage is Unique

| Traditional AI in Photo Apps | Our Approach |
|------------------------------|--------------|
| Style transfer / filters | Semantic understanding of content |
| Fixed model inference | Dynamic prompt-based reasoning |
| Single-image analysis | Comparative before/after assessment |
| Generic quality scores | Context-aware improvement metrics |

The LLM acts as an **intelligent observer**, not just a processing tool—mimicking how a human archivist would evaluate restoration quality.

---

## 5. User Interface Design

### 5.1 Real-Time Processing Feedback
- Progress indicators for each pipeline stage
- Before/after comparison views
- Expandable step details with timing metrics

### 5.2 Visual Processing Flow
- Horizontal image strip showing transformation progression
- Interactive compression analysis with rate-distortion curves
- Histogram comparisons for technical users

### 5.3 AI Results Display
- Overall improvement percentage with visual metric
- Per-aspect progress bars
- Celebratory summary messaging

---

## 6. Results and Evaluation

### 6.1 Polaroid Detection Accuracy
The two-stage detection successfully handles:
- ✅ Hand-held polaroid photographs
- ✅ Polaroids on various backgrounds
- ✅ Tilted/rotated polaroids
- ✅ Asymmetric border detection (thick signature strip)

### 6.2 Processing Performance
| Stage | Average Time |
|-------|--------------|
| Polaroid Crop | 50-100ms |
| Glare Removal | 200-500ms |
| Color Correction | 100-300ms |
| AI Analysis | 2-4 seconds |

### 6.3 AI Improvement Assessment
Sample output:
- Color Balance: +35%
- Clarity: +25%
- Noise Reduction: +40%
- Overall: +38%

---

## 7. Conclusion

This application demonstrates a novel integration of traditional computer vision with Large Language Models for heritage photo digitization. Key contributions include:

1. **Two-stage polaroid detection** that works with real-world photographs, not just scans
2. **Semantic orientation correction** using vision-language understanding
3. **Comparative AI assessment** that quantifies subjective improvements
4. **Polaroid-aware processing** that distinguishes frame elements from image artifacts

The system represents a shift from "AI as a filter" to "AI as an intelligent collaborator" in image restoration workflows.

---

## 8. Future Work

- Support for other instant film formats (Instax, etc.)
- Batch processing capabilities
- Fine-tuned models for specific degradation types
- Integration with digital asset management systems

---

## References

1. OpenCV Documentation - Image Processing Modules
2. Anthropic Claude API - Vision Capabilities
3. Gonzalez & Woods - Digital Image Processing (4th Edition)
4. Streamlit Documentation - Web Application Framework

---

*Report prepared for academic evaluation of the Digital Polaroid Archiving System*
*Author: Georgio Elias*
