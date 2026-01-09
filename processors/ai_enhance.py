"""
AI-powered image analysis and enhancement using Anthropic API.
Provides intelligent image quality assessment and recommendations.
"""

import base64
import json
import io
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AIAnalysisResult:
    """Result from AI image analysis."""
    detected_issues: List[str]
    quality_score: float
    recommendations: List[str]
    description: str
    raw_response: str
    success: bool
    error_message: Optional[str] = None


class AIEnhancer:
    """
    AI-powered image analysis using Anthropic's Claude API.
    
    Provides:
    - Intelligent issue detection
    - Subjective quality scoring
    - Enhancement recommendations
    - Image description generation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI enhancer with API key.
        
        Args:
            api_key: Anthropic API key (optional, can be set later)
        """
        self.api_key = api_key
        self.client = None
        
    def set_api_key(self, api_key: str):
        """Set the API key for Anthropic."""
        self.api_key = api_key
        self.client = None  # Reset client to use new key
    
    def _get_client(self):
        """Get or create Anthropic client."""
        if self.client is None and self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        return self.client
    
    def _encode_image(self, image: np.ndarray, quality: int = 85) -> str:
        """
        Encode image to base64 JPEG.
        
        Args:
            image: BGR numpy array
            quality: JPEG quality (0-100)
            
        Returns:
            Base64 encoded string
        """
        # Resize if too large (Claude has limits)
        max_dim = 1568  # Claude's recommended max
        h, w = image.shape[:2]
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Encode to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', image, encode_params)
        
        # Convert to base64
        return base64.standard_b64encode(buffer).decode('utf-8')
    
    def analyze_image(self, image: np.ndarray) -> AIAnalysisResult:
        """
        Analyze image using Claude for quality assessment.
        
        Args:
            image: BGR numpy array
            
        Returns:
            AIAnalysisResult with analysis details
        """
        if not self.api_key:
            return AIAnalysisResult(
                detected_issues=[],
                quality_score=0.0,
                recommendations=[],
                description="",
                raw_response="",
                success=False,
                error_message="API key not set"
            )
        
        try:
            client = self._get_client()
            
            # Encode image
            b64_image = self._encode_image(image)
            
            # Create analysis prompt
            prompt = """Analyze this scanned/digitized photograph for archival purposes.

Please evaluate:
1. Identify any quality issues (glare, color cast, fading, blur, damage, noise, perspective distortion)
2. Rate overall image quality from 1-10 (10 being perfect archival quality)
3. Suggest specific improvements that could be applied
4. Provide a brief description of the image content

Respond in valid JSON format only:
{
    "issues": ["issue1", "issue2", ...],
    "quality": 7,
    "suggestions": ["suggestion1", "suggestion2", ...],
    "description": "Brief description of image content"
}"""

            # Call API
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
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
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Parse response
            raw_text = response.content[0].text
            
            # Try to extract JSON from response
            try:
                # Handle potential markdown code blocks
                if "```json" in raw_text:
                    json_str = raw_text.split("```json")[1].split("```")[0]
                elif "```" in raw_text:
                    json_str = raw_text.split("```")[1].split("```")[0]
                else:
                    json_str = raw_text
                
                data = json.loads(json_str.strip())
                
                return AIAnalysisResult(
                    detected_issues=data.get("issues", []),
                    quality_score=float(data.get("quality", 5)),
                    recommendations=data.get("suggestions", []),
                    description=data.get("description", ""),
                    raw_response=raw_text,
                    success=True
                )
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
                return AIAnalysisResult(
                    detected_issues=[],
                    quality_score=5.0,
                    recommendations=[],
                    description=raw_text,
                    raw_response=raw_text,
                    success=True,
                    error_message="Could not parse structured response"
                )
            
        except Exception as e:
            return AIAnalysisResult(
                detected_issues=[],
                quality_score=0.0,
                recommendations=[],
                description="",
                raw_response="",
                success=False,
                error_message=str(e)
            )
    
    def compare_images(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        Compare original and processed images using AI.
        
        Args:
            original: Original BGR image
            processed: Processed BGR image
            
        Returns:
            Dictionary with comparison results
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "API key not set",
                "improvement_score": 0,
                "comparison": ""
            }
        
        try:
            client = self._get_client()
            
            # Encode both images
            b64_original = self._encode_image(original)
            b64_processed = self._encode_image(processed)
            
            prompt = """Compare these two versions of the same photograph.
The first image is the original scanned photo, and the second is the processed/enhanced version.

Please evaluate:
1. Has the processing improved the image quality? Score from -5 (worse) to +5 (much better)
2. What specific improvements are visible?
3. Are there any issues introduced by the processing?

Respond in JSON format:
{
    "improvement_score": 3,
    "improvements": ["improvement1", "improvement2"],
    "issues_introduced": ["issue1"],
    "summary": "Brief overall comparison"
}"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Original image:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_original
                            }
                        },
                        {
                            "type": "text",
                            "text": "Processed image:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_processed
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            raw_text = response.content[0].text
            
            try:
                if "```json" in raw_text:
                    json_str = raw_text.split("```json")[1].split("```")[0]
                elif "```" in raw_text:
                    json_str = raw_text.split("```")[1].split("```")[0]
                else:
                    json_str = raw_text
                
                data = json.loads(json_str.strip())
                data["success"] = True
                data["raw_response"] = raw_text
                return data
                
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "improvement_score": 0,
                    "comparison": raw_text,
                    "raw_response": raw_text
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "improvement_score": 0
            }
    
    def get_enhancement_suggestions(self, image: np.ndarray, 
                                    current_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI suggestions for optimal enhancement settings.
        
        Args:
            image: BGR input image
            current_settings: Current processing settings
            
        Returns:
            Dictionary with suggested settings adjustments
        """
        analysis = self.analyze_image(image)
        
        if not analysis.success:
            return {
                "success": False,
                "error": analysis.error_message,
                "suggestions": {}
            }
        
        suggestions = {}
        
        # Map detected issues to setting recommendations
        issues_lower = [i.lower() for i in analysis.detected_issues]
        
        if any("glare" in i or "reflection" in i for i in issues_lower):
            suggestions["glare"] = {
                "enabled": True,
                "method": "inpainting",
                "reason": "Glare/reflection detected"
            }
        
        if any("color" in i or "cast" in i or "tint" in i for i in issues_lower):
            suggestions["color"] = {
                "white_balance": "gray_world",
                "color_cast_removal": True,
                "reason": "Color cast detected"
            }
        
        if any("fade" in i or "faded" in i or "washed" in i for i in issues_lower):
            suggestions["color"] = suggestions.get("color", {})
            suggestions["color"]["saturation_boost"] = 1.3
            suggestions["color"]["reason"] = suggestions["color"].get("reason", "") + " Fading detected"
        
        if any("noise" in i or "grain" in i for i in issues_lower):
            suggestions["enhancement"] = {
                "denoise": "medium",
                "reason": "Noise/grain detected"
            }
        
        if any("blur" in i or "soft" in i for i in issues_lower):
            suggestions["enhancement"] = suggestions.get("enhancement", {})
            suggestions["enhancement"]["sharpen"] = "medium"
            suggestions["enhancement"]["reason"] = suggestions["enhancement"].get("reason", "") + " Blur detected"
        
        return {
            "success": True,
            "quality_score": analysis.quality_score,
            "detected_issues": analysis.detected_issues,
            "suggestions": suggestions,
            "description": analysis.description
        }


@dataclass
class OrientationResult:
    """Result from orientation detection."""
    needs_rotation: bool
    rotation_degrees: int  # 0, 90, 180, 270
    confidence: str
    description: str
    success: bool
    error_message: Optional[str] = None


class OrientationDetector:
    """
    AI-powered image orientation detection using Claude.
    Detects if an image is rotated and needs correction.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key."""
        self.api_key = api_key
        self.client = None
    
    def _get_client(self):
        """Get or create Anthropic client."""
        if self.client is None and self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed")
        return self.client
    
    def _encode_image(self, image: np.ndarray, quality: int = 70) -> str:
        """Encode image to base64 JPEG."""
        max_dim = 1024
        h, w = image.shape[:2]
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', image, encode_params)
        return base64.standard_b64encode(buffer).decode('utf-8')
    
    def detect_orientation(self, image: np.ndarray) -> OrientationResult:
        """
        Detect if image needs rotation correction.
        
        Args:
            image: BGR numpy array
            
        Returns:
            OrientationResult with rotation recommendation
        """
        if not self.api_key:
            return OrientationResult(
                needs_rotation=False,
                rotation_degrees=0,
                confidence="N/A",
                description="API key not available",
                success=False,
                error_message="API key not set"
            )
        
        try:
            client = self._get_client()
            b64_image = self._encode_image(image)
            
            prompt = """Analyze this scanned photograph (likely a polaroid or laminated photo) and determine if it needs rotation correction.

This is a POLAROID or physical photo being held/scanned. Key orientation clues:

MOST IMPORTANT CLUES:
1. **HAND/FINGER POSITION**: If there's a hand or fingers holding the photo, they should be at the BOTTOM of the correctly oriented image (people hold photos from below)
2. **POLAROID WHITE BORDER**: Polaroid photos have a THICKER white border at the bottom. The thicker white edge = BOTTOM of the photo
3. **The actual photo content** should have the subject matter correctly oriented

Other visual cues:
- Text should be readable left-to-right
- People's faces and bodies should be upright
- Horizon lines should be horizontal
- Buildings should be vertical
- Sky should be at the top

Determine the rotation needed to make the image correctly oriented.

Respond in JSON format only:
{
    "needs_rotation": true/false,
    "rotation_degrees": 0/90/180/270,
    "confidence": "high/medium/low",
    "description": "Brief explanation of why rotation is or isn't needed"
}

IMPORTANT: rotation_degrees should be the CLOCKWISE rotation needed to correct the image.
- 0 = image is correctly oriented (hand at bottom, thick polaroid border at bottom)
- 90 = rotate 90° clockwise to fix
- 180 = image is upside down, rotate 180° to fix
- 270 = rotate 270° clockwise (or 90° counter-clockwise) to fix"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
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
                            "text": prompt
                        }
                    ]
                }]
            )
            
            raw_text = response.content[0].text
            
            try:
                if "```json" in raw_text:
                    json_str = raw_text.split("```json")[1].split("```")[0]
                elif "```" in raw_text:
                    json_str = raw_text.split("```")[1].split("```")[0]
                else:
                    json_str = raw_text
                
                data = json.loads(json_str.strip())
                
                return OrientationResult(
                    needs_rotation=data.get("needs_rotation", False),
                    rotation_degrees=data.get("rotation_degrees", 0),
                    confidence=data.get("confidence", "low"),
                    description=data.get("description", ""),
                    success=True
                )
                
            except json.JSONDecodeError:
                return OrientationResult(
                    needs_rotation=False,
                    rotation_degrees=0,
                    confidence="low",
                    description=raw_text,
                    success=True,
                    error_message="Could not parse JSON response"
                )
                
        except Exception as e:
            return OrientationResult(
                needs_rotation=False,
                rotation_degrees=0,
                confidence="N/A",
                description="",
                success=False,
                error_message=str(e)
            )
    
    def rotate_image(self, image: np.ndarray, degrees: int) -> np.ndarray:
        """
        Rotate image by specified degrees clockwise.
        
        Args:
            image: BGR numpy array
            degrees: Rotation in degrees (0, 90, 180, 270)
            
        Returns:
            Rotated image
        """
        if degrees == 0:
            return image
        elif degrees == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif degrees == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif degrees == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # For arbitrary angles, use warpAffine
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, -degrees, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))


def ai_analyze_image(image: np.ndarray, api_key: str) -> AIAnalysisResult:
    """
    Convenience function to analyze image with AI.
    
    Args:
        image: BGR input image
        api_key: Anthropic API key
        
    Returns:
        AIAnalysisResult with analysis
    """
    enhancer = AIEnhancer(api_key)
    return enhancer.analyze_image(image)


def detect_and_fix_orientation(image: np.ndarray, api_key: str) -> tuple:
    """
    Convenience function to detect and fix image orientation.
    
    Args:
        image: BGR input image
        api_key: Anthropic API key
        
    Returns:
        Tuple of (corrected_image, OrientationResult)
    """
    detector = OrientationDetector(api_key)
    result = detector.detect_orientation(image)
    
    if result.success and result.needs_rotation:
        corrected = detector.rotate_image(image, result.rotation_degrees)
        return corrected, result
    
    return image, result

