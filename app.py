"""
Digital Image Archiving System - Streamlit Application

A comprehensive tool for digitizing and enhancing old laminated/polaroid photographs
with glare removal, perspective correction, color enhancement, and intelligent compression.
"""

import streamlit as st
import cv2
import numpy as np
import time
import io
import zipfile
from typing import Dict, Any, Optional, List, Tuple

# Import custom modules
from config import AutoConfig, DEFAULT_CONFIG, get_default_config
from processors import GlareRemover, PerspectiveCorrector, PolaroidCropper, ColorCorrector, ImageEnhancer, AIEnhancer, OrientationDetector
from compression import JPEGCompressor, calculate_metrics
from utils.image_utils import load_image, image_to_bytes, bgr_to_pil, get_image_info
from utils.visualization import (
    plot_rate_distortion_curve, 
    create_metrics_dashboard,
    create_histogram_comparison,
    create_compression_comparison_cards,
    create_processing_timeline
)

# API key from Streamlit secrets (for deployment) or environment
def get_api_key():
    """Get API key from Streamlit secrets or environment."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except:
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")

# Page configuration
st.set_page_config(
    page_title="Digital Image Archiving System",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono&display=swap');
    
    .stApp {
        font-family: 'DM Sans', sans-serif;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'DM Sans', sans-serif;
    }
    
    .sub-header {
        color: #a8a8b3;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(233, 69, 96, 0.2);
    }
    
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #e94560;
    }
    
    .metric-label {
        color: #a8a8b3;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .step-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    
    .step-badge.success {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    
    .step-badge.warning {
        background: rgba(241, 196, 15, 0.2);
        color: #f1c40f;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }
    
    .step-badge.info {
        background: rgba(52, 152, 219, 0.2);
        color: #3498db;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }
    
    .compression-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    .compression-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: #e94560;
    }
    
    .compression-card.optimal {
        border: 2px solid #e94560;
        box-shadow: 0 0 20px rgba(233, 69, 96, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.4);
    }
    
    .upload-section {
        border: 2px dashed rgba(233, 69, 96, 0.5);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(233, 69, 96, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #e94560;
        background: rgba(233, 69, 96, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #a8a8b3;
        padding: 0.5rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e94560, #ff6b6b) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {}
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = []
    if 'compression_results' not in st.session_state:
        st.session_state.compression_results = {}
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'final_image' not in st.session_state:
        st.session_state.final_image = None
    if 'auto_config' not in st.session_state:
        st.session_state.auto_config = None


def process_image(image: np.ndarray, config: Dict[str, Any], api_key: str = None) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run the complete processing pipeline on an image.
    
    Args:
        image: BGR input image
        config: Processing configuration
        api_key: Optional API key for AI-powered enhancements
        
    Returns:
        Tuple of (processed_image, pipeline_results)
    """
    pipeline_results = []
    current_image = image.copy()
    
    # Step 1: Polaroid Content Crop (FIRST - extract photo content before any processing)
    if config.get('polaroid_crop', {}).get('enabled', False):
        try:
            start_time = time.time()
            cropper = PolaroidCropper(config.get('polaroid_crop', {}))
            result = cropper.process(current_image)
            
            before_crop = current_image.copy()
            
            if result.border_detected:
                current_image = result.image
                crop_details = f"Content extracted ({result.content_ratio*100:.1f}% of original) using {result.detection_method}"
            else:
                crop_details = "No polaroid border detected - image unchanged"
            
            pipeline_results.append({
                'step': 'Polaroid Crop',
                'time': time.time() - start_time,
                'status': 'completed',
                'details': crop_details,
                'method': result.detection_method if result.border_detected else 'none',
                'before': before_crop,
                'after': current_image.copy()
            })
        except Exception as e:
            pipeline_results.append({
                'step': 'Polaroid Crop',
                'time': 0,
                'status': 'error',
                'details': str(e)
            })
    
    # Step 2: Glare Removal
    if config.get('glare', {}).get('enabled', True):
        try:
            start_time = time.time()
            glare_remover = GlareRemover(config.get('glare', {}))
            result = glare_remover.process(current_image)
            
            before_glare = current_image.copy()
            current_image = result.image
            
            pipeline_results.append({
                'step': 'Glare Removal',
                'time': time.time() - start_time,
                'status': 'completed',
                'details': f"Detected {result.regions_detected} regions ({result.coverage_percent:.1f}% coverage)",
                'method': result.method_used,
                'before': before_glare,
                'after': current_image.copy()
            })
        except Exception as e:
            pipeline_results.append({
                'step': 'Glare Removal',
                'time': 0,
                'status': 'error',
                'details': str(e)
            })
    
    # Step 3: Perspective Correction
    if config.get('perspective', {}).get('enabled', True):
        try:
            start_time = time.time()
            perspective_corrector = PerspectiveCorrector(config.get('perspective', {}))
            result = perspective_corrector.process(current_image)
            
            before_perspective = current_image.copy()
            current_image = result.image
            
            pipeline_results.append({
                'step': 'Perspective Correction',
                'time': time.time() - start_time,
                'status': 'completed',
                'details': f"Method: {result.method_used}, Confidence: {result.confidence:.2f}",
                'method': result.method_used,
                'before': before_perspective,
                'after': current_image.copy()
            })
        except Exception as e:
            pipeline_results.append({
                'step': 'Perspective Correction',
                'time': 0,
                'status': 'error',
                'details': str(e)
            })
    
    # Step 4: Color Correction
    if config.get('color', {}).get('enabled', True):
        try:
            start_time = time.time()
            color_corrector = ColorCorrector(config.get('color', {}))
            result = color_corrector.process(current_image)
            
            before_color = current_image.copy()
            current_image = result.image
            
            pipeline_results.append({
                'step': 'Color Correction',
                'time': time.time() - start_time,
                'status': 'completed',
                'details': f"WB: {result.white_balance_method}, Cast removed: {result.color_cast_corrected}",
                'method': result.white_balance_method,
                'before': before_color,
                'after': current_image.copy()
            })
        except Exception as e:
            pipeline_results.append({
                'step': 'Color Correction',
                'time': 0,
                'status': 'error',
                'details': str(e)
            })
    
    # Step 5: Enhancement (AI-powered recommendations)
    if config.get('enhancement', {}).get('enabled', True):
        try:
            start_time = time.time()
            enhancer = ImageEnhancer(config.get('enhancement', {}))
            result = enhancer.process(current_image, api_key=api_key)
            
            before_enhance = current_image.copy()
            current_image = result.image
            
            # Build details string
            details = f"Denoise: {result.denoise_applied}, Sharpen: {result.sharpen_applied}"
            if result.ai_recommendations:
                details += f" | AI: {result.ai_recommendations}"
            
            pipeline_results.append({
                'step': 'Enhancement',
                'time': time.time() - start_time,
                'status': 'completed',
                'details': details,
                'method': f"D:{result.denoise_applied} S:{result.sharpen_applied}",
                'before': before_enhance,
                'after': current_image.copy()
            })
        except Exception as e:
            pipeline_results.append({
                'step': 'Enhancement',
                'time': 0,
                'status': 'error',
                'details': str(e)
            })
    
    return current_image, pipeline_results


def run_compression_analysis(image: np.ndarray, 
                             targets: List[float]) -> Dict[float, Dict]:
    """
    Run compression analysis for multiple target sizes.
    
    Args:
        image: BGR input image
        targets: List of target sizes in KB
        
    Returns:
        Dictionary mapping target size to results
    """
    compressor = JPEGCompressor(tolerance_percent=5.0)
    results = {}
    
    for target_kb in targets:
        result = compressor.compress_to_target_size(image, target_kb)
        
        # Decode compressed image for metrics
        compressed_image = cv2.imdecode(
            np.frombuffer(result.image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        
        # Calculate metrics
        metrics = calculate_metrics(
            image, 
            compressed_image,
            compressed_size_bytes=len(result.image_bytes)
        )
        
        results[target_kb] = {
            'image_bytes': result.image_bytes,
            'quality': result.quality,
            'file_size_kb': result.file_size_kb,
            'psnr': metrics.psnr,
            'ssim': metrics.ssim,
            'mse': metrics.mse,
            'compression_ratio': metrics.compression_ratio,
            'bits_per_pixel': metrics.bits_per_pixel,
            'within_tolerance': result.within_tolerance
        }
    
    return results


def display_image_comparison(col1, col2, original: np.ndarray, processed: np.ndarray):
    """Display before/after image comparison."""
    with col1:
        st.markdown("### 📸 Original")
        st.image(bgr_to_pil(original), width='stretch')
        info = get_image_info(original)
        st.caption(f"{info['width']}×{info['height']} • {info['size_bytes']/1024:.1f} KB")
    
    with col2:
        st.markdown("### ✨ Processed")
        st.image(bgr_to_pil(processed), width='stretch')
        info = get_image_info(processed)
        st.caption(f"{info['width']}×{info['height']} • {info['size_bytes']/1024:.1f} KB")


def display_pipeline_results(pipeline_results: List[Dict]):
    """Display processing pipeline results with before/after and horizontal visualization."""
    st.markdown("### 🔄 Processing Pipeline")
    
    # Step details in expanders with before/after
    for i, result in enumerate(pipeline_results):
        status_icon = "✅" if result['status'] == 'completed' else "⚠️"
        
        with st.expander(f"{status_icon} **{result['step']}** — {result['time']*1000:.0f}ms", expanded=False):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if 'before' in result:
                    st.markdown("**Before**")
                    st.image(bgr_to_pil(result['before']), width='stretch')
            
            with col2:
                if 'after' in result:
                    st.markdown("**After**")
                    st.image(bgr_to_pil(result['after']), width='stretch')
            
            with col3:
                st.markdown("**Details**")
                st.markdown(f"Status: `{result['status']}`")
                st.markdown(f"Time: `{result['time']*1000:.1f}ms`")
                if 'method' in result:
                    st.markdown(f"Method: `{result['method']}`")
                st.caption(result.get('details', ''))
    
    # Visual pipeline - horizontal display of all steps
    st.markdown("### 📸 Visual Processing Flow")
    st.markdown("*See how your image transforms through each processing step:*")
    
    # Collect all "after" images from pipeline steps
    step_images = []
    step_labels = []
    
    # Add original as first step if we have pipeline results
    if pipeline_results and 'before' in pipeline_results[0]:
        step_images.append(pipeline_results[0]['before'])
        step_labels.append("Original")
    
    # Add each step's result
    for result in pipeline_results:
        if 'after' in result and result['status'] == 'completed':
            step_images.append(result['after'])
            step_labels.append(result['step'].replace(" Correction", "").replace(" Removal", ""))
    
    # Display horizontally
    if step_images:
        num_images = len(step_images)
        cols = st.columns(num_images)
        
        for col, img, label in zip(cols, step_images, step_labels):
            with col:
                st.image(bgr_to_pil(img), width='stretch')
                st.markdown(f"<p style='text-align: center; color: #e94560; font-weight: 600; font-size: 0.85rem;'>{label}</p>", 
                           unsafe_allow_html=True)
    
    # Add arrows between steps for visual flow
    if len(step_images) > 1:
        st.markdown("""
        <div style='text-align: center; color: #a8a8b3; font-size: 0.8rem; margin-top: -10px;'>
            ← Processing flows from left to right →
        </div>
        """, unsafe_allow_html=True)


def display_compression_results(compression_results: Dict[float, Dict], 
                                original_image: np.ndarray):
    """Display compression analysis results."""
    st.markdown("### 📊 Compression Analysis")
    
    # Info section explaining metrics
    with st.expander("ℹ️ **Understanding the Metrics** - Click to learn more", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 📈 PSNR (Peak Signal-to-Noise Ratio)
            
            **What it measures:** The difference between the original and compressed image in decibels (dB).
            
            **How to interpret:**
            - **> 40 dB**: Excellent - virtually identical to original
            - **35-40 dB**: Very Good - differences barely noticeable
            - **30-35 dB**: Good - suitable for most uses
            - **25-30 dB**: Fair - visible quality loss
            - **< 25 dB**: Poor - significant degradation
            
            **Higher is better!** ↑
            """)
        
        with col2:
            st.markdown("""
            #### 🎯 SSIM (Structural Similarity Index)
            
            **What it measures:** How similar the structure, luminance, and contrast are between images (0-1 scale).
            
            **How to interpret:**
            - **> 0.95**: Excellent - nearly perfect match
            - **0.90-0.95**: Very Good - high quality
            - **0.85-0.90**: Good - acceptable quality
            - **0.80-0.85**: Fair - noticeable differences
            - **< 0.80**: Poor - significant visual changes
            
            **Closer to 1 is better!** ↑
            """)
        
        with col3:
            st.markdown("""
            #### 📊 The Rate-Distortion Curve
            
            **What it shows:** The trade-off between file size (X-axis) and quality (Y-axis).
            
            **How to read it:**
            - **Steep rise** = big quality gains for small size increases
            - **Flat area** = diminishing returns
            - **⭐ Star** = optimal balance point
            
            **The "knee" of the curve is usually the sweet spot** - good quality without excessive file size.
            
            **Tip:** For archival, aim for SSIM > 0.90 and PSNR > 35 dB.
            """)
    
    # Prepare metrics data for chart
    metrics_data = []
    for target_kb, result in sorted(compression_results.items()):
        metrics_data.append({
            'size_kb': result['file_size_kb'],
            'psnr': result['psnr'],
            'ssim': result['ssim'],
            'quality': result['quality']
        })
    
    # Rate-distortion curve
    fig = plot_rate_distortion_curve(metrics_data, "Quality vs. File Size")
    st.plotly_chart(fig, width='stretch')
    
    # Compression cards
    st.markdown("#### Compression Options")
    
    # Find optimal point
    efficiencies = []
    for target_kb, result in compression_results.items():
        psnr_norm = min(result['psnr'] / 50, 1.0) if result['psnr'] != float('inf') else 1.0
        quality_score = psnr_norm * 0.4 + result['ssim'] * 0.6
        size_factor = np.log10(result['file_size_kb'] + 1) / np.log10(1001)
        efficiency = quality_score / (size_factor + 0.1)
        efficiencies.append((target_kb, efficiency))
    
    optimal_target = max(efficiencies, key=lambda x: x[1])[0]
    
    cols = st.columns(len(compression_results))
    
    for col, (target_kb, result) in zip(cols, sorted(compression_results.items())):
        is_optimal = (target_kb == optimal_target)
        
        with col:
            # Card styling
            card_class = "optimal" if is_optimal else ""
            
            st.markdown(f"""
            <div class="compression-card {card_class}">
                <h4 style="color: {'#e94560' if is_optimal else '#fff'}; margin-bottom: 0.5rem;">
                    {target_kb:.0f} KB {'⭐' if is_optimal else ''}
                </h4>
                <p style="color: #a8a8b3; font-size: 0.9rem; margin-bottom: 0.3rem;">
                    Actual: {result['file_size_kb']:.1f} KB
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            psnr_display = f"{result['psnr']:.1f}" if result['psnr'] != float('inf') else "∞"
            st.metric("PSNR", f"{psnr_display} dB")
            st.metric("SSIM", f"{result['ssim']:.4f}")
            st.metric("Quality", f"{result['quality']}%")
            
            # Download button
            st.download_button(
                label=f"📥 Download",
                data=result['image_bytes'],
                file_name=f"photo_{target_kb:.0f}kb.jpg",
                mime="image/jpeg",
                key=f"download_{target_kb}"
            )
    
    # Recommendation
    optimal_result = compression_results[optimal_target]
    st.info(f"""
    💡 **Recommendation**: The **{optimal_target:.0f} KB** version offers the optimal balance between 
    quality (PSNR: {optimal_result['psnr']:.1f} dB, SSIM: {optimal_result['ssim']:.4f}) and file size.
    """)


def create_download_package(original: np.ndarray, 
                           processed: np.ndarray,
                           compression_results: Dict[float, Dict],
                           pipeline_results: List[Dict]) -> bytes:
    """Create a ZIP package with all versions and a report."""
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Original image
        _, orig_buf = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 95])
        zf.writestr("original.jpg", orig_buf.tobytes())
        
        # Processed full quality
        _, proc_buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        zf.writestr("processed_full_quality.jpg", proc_buf.tobytes())
        
        # Compressed versions
        for target_kb, result in compression_results.items():
            zf.writestr(f"compressed_{target_kb:.0f}kb.jpg", result['image_bytes'])
        
        # Create report
        report = generate_report(pipeline_results, compression_results)
        zf.writestr("processing_report.txt", report)
    
    buffer.seek(0)
    return buffer.getvalue()


def generate_report(pipeline_results: List[Dict], 
                   compression_results: Dict[float, Dict]) -> str:
    """Generate a text report of the processing."""
    lines = [
        "=" * 60,
        "DIGITAL IMAGE ARCHIVING SYSTEM - PROCESSING REPORT",
        "=" * 60,
        "",
        "PROCESSING PIPELINE",
        "-" * 40,
    ]
    
    total_time = 0
    for result in pipeline_results:
        status = "✓" if result['status'] == 'completed' else "✗"
        lines.append(f"{status} {result['step']}: {result['time']*1000:.1f}ms")
        lines.append(f"   {result.get('details', '')}")
        total_time += result['time']
    
    lines.extend([
        "",
        f"Total processing time: {total_time*1000:.1f}ms",
        "",
        "COMPRESSION ANALYSIS",
        "-" * 40,
    ])
    
    for target_kb, result in sorted(compression_results.items()):
        lines.append(f"\n{target_kb:.0f} KB Target:")
        lines.append(f"  Actual size: {result['file_size_kb']:.1f} KB")
        lines.append(f"  JPEG Quality: {result['quality']}%")
        lines.append(f"  PSNR: {result['psnr']:.2f} dB")
        lines.append(f"  SSIM: {result['ssim']:.4f}")
        lines.append(f"  Compression ratio: {result['compression_ratio']:.1f}x")
    
    lines.extend([
        "",
        "=" * 60,
        "Generated by Digital Image Archiving System",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">Digital Image Archiving System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your old photographs into pristine digital archives</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # API key from Streamlit secrets
        api_key = get_api_key()
        
        # AI-powered orientation detection
        enable_orientation = st.checkbox("🔄 AI Orientation Detection", value=True,
                                         help="Use Claude AI to detect and correct image rotation")
        
        st.markdown("---")
        st.markdown("### Processing Options")
        
        enable_glare = st.checkbox("Glare Removal", value=True)
        enable_perspective = st.checkbox("Perspective Correction", value=True)
        enable_color = st.checkbox("Color Correction", value=True)
        enable_enhance = st.checkbox("Enhancement", value=True)
        
        st.markdown("---")
        st.markdown("### Polaroid Options")
        enable_polaroid_crop = st.checkbox("📷 Crop Polaroid Content", value=False,
                                           help="Extract the photo content from inside the polaroid frame (removes white border)")
        
        st.markdown("---")
        st.markdown("### Compression Targets")
        
        targets = st.multiselect(
            "Target sizes (KB)",
            options=[30, 50, 100, 200, 500, 1000],
            default=[30, 100, 500, 1000]
        )
        
        st.markdown("---")
        
        # Advanced settings in expander
        with st.expander("🔧 Advanced Settings"):
            st.markdown("#### Glare Removal")
            glare_method = st.selectbox("Method", ["auto", "adaptive", "inpainting"], index=0)
            inpaint_radius = st.slider("Inpaint Radius", 1, 15, 5)
            
            st.markdown("#### Color Correction")
            wb_method = st.selectbox("White Balance", ["auto", "gray_world", "white_patch", "combined"], index=0)
            saturation = st.slider("Saturation Boost", 0.8, 1.5, 1.1, 0.05)
            
            st.markdown("#### Enhancement")
            denoise = st.selectbox("Denoise", ["auto", "off", "light", "medium", "strong"], index=0)
            sharpen = st.selectbox("Sharpen", ["auto", "off", "light", "medium", "strong"], index=0)
    
    # Main content area
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a photograph to digitize",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
        help="Supported formats: JPG, PNG, BMP, TIFF, WebP"
    )
    
    if uploaded_file is not None:
        # Load image
        try:
            image_bytes = uploaded_file.read()
            original_image = load_image(image_bytes)
            st.session_state.original_image = original_image
            
            # Auto-analyze image
            with st.spinner("🔍 Analyzing image..."):
                auto_config = AutoConfig(original_image)
                st.session_state.auto_config = auto_config
                analysis = auto_config.analysis
            
            # Display analysis summary
            st.markdown("### 📋 Image Analysis")
            
            # Info expander for analysis metrics
            with st.expander("ℹ️ **What do these metrics mean?** - Click to learn", expanded=False):
                st.markdown("""
                | Metric | Description | Ideal Range |
                |--------|-------------|-------------|
                | **Brightness** | Overall lightness of the image (0=black, 1=white) | 0.4 - 0.6 |
                | **Contrast** | Difference between light and dark areas (0=flat, 1=high contrast) | 0.3 - 0.7 |
                | **Noise Level** | Amount of grain/noise detected (0=clean, 1=very noisy) | < 0.3 |
                | **Glare** | Specular highlights from lamination (🔆 = detected, ✓ = none) | 0 (none) |
                | **Faded** | Color saturation loss typical in old photos (📉 = yes, ✓ = no) | Not faded |
                
                **The system automatically adjusts processing based on these values!**
                """)
            
            analysis_cols = st.columns(5)
            with analysis_cols[0]:
                st.metric("Brightness", f"{analysis.brightness:.2f}",
                         help="0 = very dark, 1 = very bright. Ideal: 0.4-0.6")
            with analysis_cols[1]:
                st.metric("Contrast", f"{analysis.contrast:.2f}",
                         help="0 = flat/no contrast, 1 = high contrast. Ideal: 0.3-0.7")
            with analysis_cols[2]:
                st.metric("Noise Level", f"{analysis.noise_level:.2f}",
                         help="0 = clean image, 1 = very noisy. Below 0.3 is good")
            with analysis_cols[3]:
                glare_icon = "🔆" if analysis.has_glare else "✓"
                st.metric("Glare Detected", f"{glare_icon} {analysis.glare_severity:.2f}",
                         help="Glare severity from lamination. 🔆 = glare detected, ✓ = no glare")
            with analysis_cols[4]:
                faded_icon = "📉" if analysis.is_faded else "✓"
                st.metric("Faded", faded_icon,
                         help="📉 = colors appear faded/washed out, ✓ = colors are vibrant")
            
            st.markdown("---")
            
            # Process button
            if st.button("🚀 Process Image", type="primary", width='stretch'):
                # Process image
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Store the working image (may be rotated)
                working_image = original_image.copy()
                orientation_result = None
                
                # Step 0: AI Orientation Detection
                if enable_orientation and api_key:
                    status_text.text("🔄 Detecting image orientation with AI...")
                    progress_bar.progress(5)
                    
                    try:
                        detector = OrientationDetector(api_key)
                        orientation_result = detector.detect_orientation(working_image)
                        
                        if orientation_result.success and orientation_result.needs_rotation:
                            working_image = detector.rotate_image(working_image, orientation_result.rotation_degrees)
                            st.session_state.orientation_result = orientation_result
                            st.info(f"🔄 **Orientation Corrected**: Rotated {orientation_result.rotation_degrees}° clockwise. {orientation_result.description}")
                        elif orientation_result.success:
                            st.session_state.orientation_result = orientation_result
                    except Exception as e:
                        st.warning(f"Orientation detection failed: {e}")
                
                # Re-analyze with corrected orientation
                auto_config = AutoConfig(working_image)
                st.session_state.auto_config = auto_config
                
                # Build configuration
                config = {
                    'glare': {
                        **auto_config.get_glare_config(),
                        'enabled': enable_glare,
                        'method': glare_method if glare_method != 'auto' else auto_config.get_glare_config().get('method', 'adaptive'),
                        'inpaint_radius': inpaint_radius,
                    },
                    'perspective': {
                        **auto_config.get_perspective_config(),
                        'enabled': enable_perspective,
                    },
                    'color': {
                        **auto_config.get_color_config(),
                        'enabled': enable_color,
                        'white_balance_method': wb_method if wb_method != 'auto' else auto_config.get_color_config().get('white_balance_method', 'combined'),
                        'saturation_boost': saturation,
                    },
                    'enhancement': {
                        **auto_config.get_enhancement_config(),
                        'enabled': enable_enhance,
                        'denoise': denoise if denoise != 'auto' else auto_config.get_enhancement_config().get('denoise', 'auto'),
                        'sharpen': sharpen if sharpen != 'auto' else auto_config.get_enhancement_config().get('sharpen', 'auto'),
                    },
                    'polaroid_crop': {
                        'enabled': enable_polaroid_crop,
                    }
                }
                
                status_text.text("Processing image...")
                progress_bar.progress(15)
                
                processed_image, pipeline_results = process_image(working_image, config, api_key=api_key)
                
                # Add orientation step to pipeline results if rotation was applied
                if orientation_result and orientation_result.needs_rotation:
                    pipeline_results.insert(0, {
                        'step': 'Orientation Correction',
                        'time': 0.5,
                        'status': 'completed',
                        'details': f"Rotated {orientation_result.rotation_degrees}° ({orientation_result.confidence} confidence)",
                        'method': 'AI Detection',
                        'before': original_image.copy(),
                        'after': working_image.copy()
                    })
                st.session_state.final_image = processed_image
                st.session_state.pipeline_results = pipeline_results
                
                progress_bar.progress(50)
                status_text.text("Running compression analysis...")
                
                # Compression analysis
                if targets:
                    compression_results = run_compression_analysis(processed_image, targets)
                    st.session_state.compression_results = compression_results
                
                progress_bar.progress(90)
                
                # AI Comparison Analysis (if API key provided)
                if api_key:
                    status_text.text("Running AI comparison analysis...")
                    try:
                        ai_enhancer = AIEnhancer(api_key)
                        # Compare original vs processed to show improvements
                        comparison_result = ai_enhancer.compare_images(original_image, processed_image)
                        if comparison_result.success:
                            st.session_state.ai_comparison = comparison_result
                    except Exception as e:
                        st.warning(f"AI comparison failed: {e}")
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.success("✨ Image processing complete!")
            
            # Display results if available
            if st.session_state.final_image is not None:
                st.markdown("---")
                
                # Image comparison
                col1, col2 = st.columns(2)
                display_image_comparison(col1, col2, original_image, st.session_state.final_image)
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Compression", "🔄 Pipeline Details", "📈 Histograms"])
                
                with tab1:
                    if st.session_state.compression_results:
                        display_compression_results(
                            st.session_state.compression_results,
                            st.session_state.final_image
                        )
                
                with tab2:
                    if st.session_state.pipeline_results:
                        display_pipeline_results(st.session_state.pipeline_results)
                        
                        # Processing timeline
                        fig = create_processing_timeline(st.session_state.pipeline_results)
                        st.plotly_chart(fig, width='stretch')
                
                with tab3:
                    fig = create_histogram_comparison(original_image, st.session_state.final_image)
                    st.plotly_chart(fig, width='stretch')
                
                # AI Comparison Results - Show Improvements
                if hasattr(st.session_state, 'ai_comparison') and st.session_state.ai_comparison:
                    comparison = st.session_state.ai_comparison
                    
                    if comparison.success and comparison.improvements:
                        st.markdown("---")
                        st.markdown("### ✨ AI Analysis: Improvements Made")
                        
                        # Overall improvement metric
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(
                                "Overall Improvement", 
                                f"+{comparison.overall_improvement:.0f}%",
                                delta=f"{comparison.overall_improvement:.0f}% better"
                            )
                        
                        with col2:
                            # Summary
                            st.success(f"🎉 {comparison.summary}")
                        
                        # Individual improvements with progress bars
                        st.markdown("#### 📈 Detailed Improvements")
                        
                        for imp in comparison.improvements:
                            aspect = imp.get('aspect', 'Quality')
                            percent = imp.get('percent', 0)
                            description = imp.get('description', '')
                            
                            col1, col2, col3 = st.columns([2, 1, 3])
                            with col1:
                                st.markdown(f"**{aspect}**")
                            with col2:
                                st.markdown(f"<span style='color: #4CAF50; font-weight: bold;'>+{percent}%</span>", unsafe_allow_html=True)
                            with col3:
                                st.progress(min(percent / 100, 1.0))
                            
                            if description:
                                st.caption(f"↳ {description}")
                
                # Download all
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    # Download processed full quality
                    proc_bytes = image_to_bytes(st.session_state.final_image, quality=95)
                    st.download_button(
                        label="📥 Download Processed (Full Quality)",
                        data=proc_bytes,
                        file_name="processed_full.jpg",
                        mime="image/jpeg",
                        width='stretch'
                    )
                
                with col2:
                    # Download package
                    if st.session_state.compression_results:
                        package = create_download_package(
                            original_image,
                            st.session_state.final_image,
                            st.session_state.compression_results,
                            st.session_state.pipeline_results
                        )
                        st.download_button(
                            label="📦 Download All Versions (ZIP)",
                            data=package,
                            file_name="photo_archive.zip",
                            mime="application/zip",
                            width='stretch'
                        )
                
                with col3:
                    # Generate report
                    if st.session_state.compression_results:
                        report = generate_report(
                            st.session_state.pipeline_results,
                            st.session_state.compression_results
                        )
                        st.download_button(
                            label="📄 Download Report",
                            data=report,
                            file_name="processing_report.txt",
                            mime="text/plain",
                            width='stretch'
                        )
        
        except Exception as e:
            st.error(f"Error loading image: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # Show placeholder/instructions when no image uploaded
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: #e94560; margin-bottom: 1rem;">📤 Upload Your Photo</h3>
            <p style="color: #a8a8b3;">
                Drag and drop an image or click to browse.<br>
                Supported formats: JPG, PNG, BMP, TIFF, WebP
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Features overview
        st.markdown("### ✨ Features")
        
        features = st.columns(4)
        with features[0]:
            st.markdown("""
            #### 🔆 Glare Removal
            Automatically detects and removes reflections from laminated photos
            """)
        
        with features[1]:
            st.markdown("""
            #### 📐 Perspective Correction
            Straightens tilted or skewed photographs
            """)
        
        with features[2]:
            st.markdown("""
            #### 🎨 Color Correction
            Restores faded colors and removes color casts
            """)
        
        with features[3]:
            st.markdown("""
            #### 📊 Smart Compression
            Optimal quality at target file sizes
            """)


if __name__ == "__main__":
    main()

