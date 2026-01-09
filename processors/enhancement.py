"""
Simple image enhancement module with AI-powered recommendations.
Since color correction already handles most improvements, this module
provides optional light touch-ups based on AI analysis.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import base64
import json


@dataclass 
class EnhancementResult:
    """Result from image enhancement processing."""
    image: np.ndarray
    denoise_applied: str
    sharpen_applied: str
    ai_recommendations: str
    processing_time: float


class ImageEnhancer:
    """
    Simple image enhancer with AI-powered recommendations.
    
    The color correction step already handles most improvements.
    This module provides optional light denoising and sharpening
    based on AI analysis of the image.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_config = {
            "denoise": "auto",
            "sharpen": "auto",
            "use_ai": True,
        }
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def get_ai_recommendations(self, image: np.ndarray, api_key: str) -> Dict[str, Any]:
        """
        Use Claude AI to analyze image and recommend enhancement settings.
        
        Returns dict with:
        - denoise: "none", "light", "medium"
        - sharpen: "none", "light", "medium" 
        - reason: explanation
        """
        try:
            import anthropic
            
            # Convert image to JPEG bytes
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = bytes(buffer)
            b64_image = base64.b64encode(image_bytes).decode()
            
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": """Analyze this photograph and recommend enhancement settings.
The image has already been color corrected, so focus ONLY on:
1. Noise level - is denoising needed?
2. Sharpness - is sharpening needed?

Be conservative - only recommend if clearly needed.
Most photos look fine after color correction.

Respond in JSON format only:
{"denoise": "none|light|medium", "sharpen": "none|light|medium", "reason": "brief explanation"}"""
                        }
                    ]
                }]
            )
            
            # Parse response
            text = response.content[0].text
            # Extract JSON from response
            if '{' in text and '}' in text:
                json_str = text[text.find('{'):text.rfind('}')+1]
                return json.loads(json_str)
            
            return {"denoise": "none", "sharpen": "none", "reason": "Could not parse AI response"}
            
        except Exception as e:
            return {"denoise": "none", "sharpen": "none", "reason": f"AI unavailable: {str(e)[:50]}"}
    
    def estimate_noise_level(self, image: np.ndarray) -> float:
        """Simple noise estimation using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize: typical clean image ~100-500, noisy ~500+
        noise_level = min(max(variance - 100, 0) / 1000.0, 1.0)
        return float(noise_level)
    
    def measure_sharpness(self, image: np.ndarray) -> float:
        """Simple sharpness measurement."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return min(laplacian.var() / 1000.0, 1.0)
    
    def denoise_light(self, image: np.ndarray) -> np.ndarray:
        """Light denoising - gentle touch up."""
        return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    
    def denoise_medium(self, image: np.ndarray) -> np.ndarray:
        """Medium denoising with edge preservation."""
        bilateral = cv2.bilateralFilter(image, 7, 50, 50)
        return cv2.fastNlMeansDenoisingColored(bilateral, None, 4, 4, 7, 21)
    
    def sharpen_light(self, image: np.ndarray) -> np.ndarray:
        """Light sharpening using unsharp mask."""
        blurred = cv2.GaussianBlur(image, (3, 3), 1.0)
        return cv2.addWeighted(image, 1.3, blurred, -0.3, 0)
    
    def sharpen_medium(self, image: np.ndarray) -> np.ndarray:
        """Medium sharpening."""
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def process(self, image: np.ndarray, api_key: str = None) -> EnhancementResult:
        """
        Process image with optional AI recommendations.
        
        Args:
            image: BGR input image
            api_key: Optional Anthropic API key for AI recommendations
            
        Returns:
            EnhancementResult
        """
        import time
        start_time = time.time()
        
        result = image.copy()
        denoise_applied = "none"
        sharpen_applied = "none"
        ai_recommendations = ""
        
        # Get AI recommendations if API key provided
        if api_key and self.config.get("use_ai", True):
            ai_result = self.get_ai_recommendations(image, api_key)
            ai_recommendations = ai_result.get("reason", "")
            
            # Apply AI-recommended denoising
            denoise_rec = ai_result.get("denoise", "none")
            if denoise_rec == "light":
                result = self.denoise_light(result)
                denoise_applied = "light"
            elif denoise_rec == "medium":
                result = self.denoise_medium(result)
                denoise_applied = "medium"
            
            # Apply AI-recommended sharpening
            sharpen_rec = ai_result.get("sharpen", "none")
            if sharpen_rec == "light":
                result = self.sharpen_light(result)
                sharpen_applied = "light"
            elif sharpen_rec == "medium":
                result = self.sharpen_medium(result)
                sharpen_applied = "medium"
        
        else:
            # Simple auto mode without AI
            denoise_mode = self.config.get("denoise", "auto")
            sharpen_mode = self.config.get("sharpen", "auto")
            
            if denoise_mode == "auto":
                noise = self.estimate_noise_level(image)
                if noise > 0.4:
                    result = self.denoise_medium(result)
                    denoise_applied = "medium"
                elif noise > 0.2:
                    result = self.denoise_light(result)
                    denoise_applied = "light"
                ai_recommendations = f"Auto: noise level {noise:.2f}"
            elif denoise_mode == "light":
                result = self.denoise_light(result)
                denoise_applied = "light"
            elif denoise_mode == "medium":
                result = self.denoise_medium(result)
                denoise_applied = "medium"
            
            if sharpen_mode == "auto":
                sharpness = self.measure_sharpness(result)
                if sharpness < 0.3:
                    result = self.sharpen_light(result)
                    sharpen_applied = "light"
                ai_recommendations += f", sharpness {sharpness:.2f}"
            elif sharpen_mode == "light":
                result = self.sharpen_light(result)
                sharpen_applied = "light"
            elif sharpen_mode == "medium":
                result = self.sharpen_medium(result)
                sharpen_applied = "medium"
        
        return EnhancementResult(
            image=result,
            denoise_applied=denoise_applied,
            sharpen_applied=sharpen_applied,
            ai_recommendations=ai_recommendations,
            processing_time=time.time() - start_time
        )


def enhance_image(image: np.ndarray, config: Optional[Dict] = None, api_key: str = None) -> np.ndarray:
    """
    Convenience function to enhance an image.
    """
    enhancer = ImageEnhancer(config)
    result = enhancer.process(image, api_key)
    return result.image
