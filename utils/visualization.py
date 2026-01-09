"""
Visualization utilities for image comparison and metrics display.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_comparison_slider(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side comparison image.
    
    Args:
        before: Before image (BGR)
        after: After image (BGR)
        
    Returns:
        Combined comparison image
    """
    # Ensure same dimensions
    h1, w1 = before.shape[:2]
    h2, w2 = after.shape[:2]
    
    # Resize to match if needed
    if (h1, w1) != (h2, w2):
        # Use the smaller dimensions
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        before = cv2.resize(before, (target_w, target_h))
        after = cv2.resize(after, (target_w, target_h))
    
    # Create side by side
    combined = np.hstack([before, after])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Before", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "After", (w1 + 10, 30), font, 1, (255, 255, 255), 2)
    
    return combined


def create_side_by_side(images: List[np.ndarray], 
                        labels: List[str],
                        max_height: int = 400) -> np.ndarray:
    """
    Create side-by-side comparison of multiple images.
    
    Args:
        images: List of images (BGR)
        labels: List of labels for each image
        max_height: Maximum height of output
        
    Returns:
        Combined comparison image
    """
    if len(images) != len(labels):
        raise ValueError("Number of images must match number of labels")
    
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize all images to same height
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = max_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, max_height)))
    
    # Combine horizontally
    combined = np.hstack(resized)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_offset = 0
    for i, (img, label) in enumerate(zip(resized, labels)):
        cv2.putText(combined, label, (x_offset + 10, 30), font, 0.7, (255, 255, 255), 2)
        x_offset += img.shape[1]
    
    return combined


def plot_rate_distortion_curve(metrics_data: List[Dict[str, Any]],
                               title: str = "Rate-Distortion Curve") -> go.Figure:
    """
    Create interactive Plotly rate-distortion curve.
    
    Args:
        metrics_data: List of dicts with 'size_kb', 'psnr', 'ssim', 'quality'
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Extract data
    sizes = [m['size_kb'] for m in metrics_data]
    psnrs = [m['psnr'] if m['psnr'] != float('inf') else 60 for m in metrics_data]
    ssims = [m['ssim'] for m in metrics_data]
    qualities = [m.get('quality', 0) for m in metrics_data]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add PSNR trace
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=psnrs,
            name="PSNR (dB)",
            mode="lines+markers",
            marker=dict(size=10, color="#2ecc71"),
            line=dict(width=2, color="#2ecc71"),
            hovertemplate="Size: %{x:.1f} KB<br>PSNR: %{y:.2f} dB<extra></extra>"
        ),
        secondary_y=False,
    )
    
    # Add SSIM trace
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=ssims,
            name="SSIM",
            mode="lines+markers",
            marker=dict(size=10, color="#3498db"),
            line=dict(width=2, color="#3498db"),
            hovertemplate="Size: %{x:.1f} KB<br>SSIM: %{y:.4f}<extra></extra>"
        ),
        secondary_y=True,
    )
    
    # Find optimal point (highest efficiency)
    # Efficiency = (PSNR/50 * 0.4 + SSIM * 0.6) / log(size)
    efficiencies = []
    for i in range(len(sizes)):
        psnr_norm = min(psnrs[i] / 50, 1.0)
        quality_score = psnr_norm * 0.4 + ssims[i] * 0.6
        size_factor = np.log10(sizes[i] + 1) / np.log10(1001)
        efficiency = quality_score / (size_factor + 0.1)
        efficiencies.append(efficiency)
    
    optimal_idx = np.argmax(efficiencies)
    
    # Add optimal point annotation
    fig.add_trace(
        go.Scatter(
            x=[sizes[optimal_idx]],
            y=[psnrs[optimal_idx]],
            name="Optimal",
            mode="markers",
            marker=dict(size=20, color="#e74c3c", symbol="star"),
            hovertemplate=f"Optimal Point<br>Size: {sizes[optimal_idx]:.1f} KB<br>PSNR: {psnrs[optimal_idx]:.2f} dB<br>SSIM: {ssims[optimal_idx]:.4f}<extra></extra>"
        ),
        secondary_y=False,
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color="#2c3e50")
        ),
        xaxis_title="File Size (KB)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    fig.update_yaxes(
        title_text="PSNR (dB)",
        secondary_y=False,
        gridcolor="#ecf0f1",
        range=[20, max(psnrs) + 5]
    )
    fig.update_yaxes(
        title_text="SSIM",
        secondary_y=True,
        gridcolor="#ecf0f1",
        range=[min(ssims) - 0.05, 1.0]
    )
    fig.update_xaxes(
        gridcolor="#ecf0f1",
        type="log" if max(sizes) > 500 else "linear"
    )
    
    return fig


def create_metrics_dashboard(metrics: Dict[str, Any]) -> go.Figure:
    """
    Create visual dashboard for quality metrics.
    
    Args:
        metrics: Dictionary with PSNR, SSIM, MSE values
        
    Returns:
        Plotly figure
    """
    # Create gauge charts for key metrics
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("PSNR (dB)", "SSIM Index", "Quality Score")
    )
    
    # PSNR gauge
    psnr = metrics.get('psnr', 0)
    psnr_display = min(psnr, 50) if psnr != float('inf') else 50
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=psnr_display,
            number={"suffix": " dB", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 50]},
                "bar": {"color": "#2ecc71"},
                "steps": [
                    {"range": [0, 25], "color": "#e74c3c"},
                    {"range": [25, 35], "color": "#f1c40f"},
                    {"range": [35, 50], "color": "#2ecc71"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 4},
                    "thickness": 0.75,
                    "value": 35
                }
            }
        ),
        row=1, col=1
    )
    
    # SSIM gauge
    ssim = metrics.get('ssim', 0)
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=ssim,
            number={"font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#3498db"},
                "steps": [
                    {"range": [0, 0.8], "color": "#e74c3c"},
                    {"range": [0.8, 0.9], "color": "#f1c40f"},
                    {"range": [0.9, 1], "color": "#2ecc71"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 4},
                    "thickness": 0.75,
                    "value": 0.9
                }
            }
        ),
        row=1, col=2
    )
    
    # Overall quality score
    psnr_norm = min(psnr / 50, 1.0) if psnr != float('inf') else 1.0
    quality_score = (psnr_norm * 0.4 + ssim * 0.6) * 10
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=quality_score,
            number={"suffix": "/10", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "#9b59b6"},
                "steps": [
                    {"range": [0, 5], "color": "#e74c3c"},
                    {"range": [5, 7], "color": "#f1c40f"},
                    {"range": [7, 10], "color": "#2ecc71"},
                ],
                "threshold": {
                    "line": {"color": "#2c3e50", "width": 4},
                    "thickness": 0.75,
                    "value": 7
                }
            }
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
        template="plotly_white"
    )
    
    return fig


def create_histogram_comparison(original: np.ndarray, 
                                processed: np.ndarray) -> go.Figure:
    """
    Create histogram comparison between original and processed images.
    
    Args:
        original: Original image (BGR)
        processed: Processed image (BGR)
        
    Returns:
        Plotly figure
    """
    # Convert to grayscale for luminance histogram
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256]).flatten()
    hist_proc = cv2.calcHist([proc_gray], [0], None, [256], [0, 256]).flatten()
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=list(range(256)),
            y=hist_orig,
            name="Original",
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.3)',
            line=dict(color='#e74c3c', width=1),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(256)),
            y=hist_proc,
            name="Processed",
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.3)',
            line=dict(color='#2ecc71', width=1),
        )
    )
    
    fig.update_layout(
        title="Luminance Histogram Comparison",
        xaxis_title="Pixel Value",
        yaxis_title="Frequency",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


def create_compression_comparison_cards(compression_results: Dict[float, Dict]) -> List[Dict]:
    """
    Create data for compression comparison cards.
    
    Args:
        compression_results: Dictionary mapping target KB to results
        
    Returns:
        List of card data dictionaries
    """
    cards = []
    
    for target_kb, result in sorted(compression_results.items()):
        psnr = result.get('psnr', 0)
        ssim = result.get('ssim', 0)
        actual_size = result.get('file_size_kb', target_kb)
        quality = result.get('quality', 0)
        
        # Determine quality rating
        if psnr > 40 or ssim > 0.95:
            rating = "Excellent"
            color = "#2ecc71"
        elif psnr > 35 or ssim > 0.90:
            rating = "Very Good"
            color = "#27ae60"
        elif psnr > 30 or ssim > 0.85:
            rating = "Good"
            color = "#f1c40f"
        elif psnr > 25 or ssim > 0.80:
            rating = "Fair"
            color = "#e67e22"
        else:
            rating = "Poor"
            color = "#e74c3c"
        
        cards.append({
            "target_kb": target_kb,
            "actual_size_kb": round(actual_size, 1),
            "quality": quality,
            "psnr": round(psnr, 2) if psnr != float('inf') else "∞",
            "ssim": round(ssim, 4),
            "rating": rating,
            "color": color,
        })
    
    return cards


def create_processing_timeline(pipeline_results: List[Dict]) -> go.Figure:
    """
    Create a timeline visualization of processing steps.
    
    Args:
        pipeline_results: List of dicts with 'step', 'time', 'status'
        
    Returns:
        Plotly figure
    """
    steps = [r['step'] for r in pipeline_results]
    times = [r['time'] * 1000 for r in pipeline_results]  # Convert to ms
    statuses = [r.get('status', 'completed') for r in pipeline_results]
    
    colors = ['#2ecc71' if s == 'completed' else '#e74c3c' for s in statuses]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            y=steps,
            x=times,
            orientation='h',
            marker_color=colors,
            text=[f"{t:.1f}ms" for t in times],
            textposition='outside',
            hovertemplate="%{y}<br>%{x:.1f}ms<extra></extra>"
        )
    )
    
    fig.update_layout(
        title="Processing Pipeline Timeline",
        xaxis_title="Processing Time (ms)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=50 + len(steps) * 40,
        margin=dict(l=150, r=50, t=50, b=50),
    )
    
    return fig


def create_before_after_grid(original: np.ndarray,
                            steps: Dict[str, np.ndarray],
                            max_size: int = 300) -> np.ndarray:
    """
    Create a grid showing original and each processing step.
    
    Args:
        original: Original image
        steps: Dictionary mapping step name to resulting image
        max_size: Maximum dimension for thumbnails
        
    Returns:
        Grid image
    """
    def resize_to_thumbnail(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    
    # Create thumbnails
    thumbnails = [("Original", resize_to_thumbnail(original))]
    for name, img in steps.items():
        thumbnails.append((name, resize_to_thumbnail(img)))
    
    # Calculate grid dimensions
    n = len(thumbnails)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    
    # Create canvas
    thumb_h = thumbnails[0][1].shape[0]
    thumb_w = thumbnails[0][1].shape[1]
    label_height = 30
    
    canvas_h = rows * (thumb_h + label_height)
    canvas_w = cols * thumb_w
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas.fill(40)  # Dark background
    
    # Place thumbnails
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (name, thumb) in enumerate(thumbnails):
        row = idx // cols
        col = idx % cols
        
        y = row * (thumb_h + label_height) + label_height
        x = col * thumb_w
        
        # Resize thumbnail to fit if needed
        th, tw = thumb.shape[:2]
        if th != thumb_h or tw != thumb_w:
            thumb = cv2.resize(thumb, (thumb_w, thumb_h))
        
        canvas[y:y+thumb_h, x:x+thumb_w] = thumb
        
        # Add label
        text_size = cv2.getTextSize(name, font, 0.5, 1)[0]
        text_x = x + (thumb_w - text_size[0]) // 2
        cv2.putText(canvas, name, (text_x, y - 10), font, 0.5, (255, 255, 255), 1)
    
    return canvas

