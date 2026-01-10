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


@dataclass
class AIComparisonResult:
    """Result from AI comparison of original vs processed image."""
    improvements: List[Dict[str, Any]]  # List of {aspect, improvement_percent, description}
    overall_improvement: float  # Percentage
    summary: str  # Positive summary of changes
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
    
    def compare_images(self, original: np.ndarray, processed: np.ndarray) -> AIComparisonResult:
        """
        Compare original and processed images - focus on POSITIVE improvements only.
        
        Args:
            original: Original BGR image
            processed: Processed BGR image
            
        Returns:
            AIComparisonResult with improvements and percentages
        """
        if not self.api_key:
            return AIComparisonResult(
                improvements=[],
                overall_improvement=0,
                summary="API key not available",
                success=False,
                error_message="API key not set"
            )
        
        try:
            client = self._get_client()
            
            # Encode both images
            b64_original = self._encode_image(original)
            b64_processed = self._encode_image(processed)
            
            prompt = """Compare the ORIGINAL image (first) with the PROCESSED image (second).

Your task is to identify and celebrate the IMPROVEMENTS made. Focus ONLY on positive changes.
For each improvement, estimate the percentage improvement (10-100%).

Evaluate these aspects:
- Color accuracy (white balance, vibrancy, saturation)
- Clarity and sharpness
- Noise reduction
- Contrast and exposure
- Glare/reflection removal
- Overall visual appeal

IMPORTANT: Be enthusiastic and positive! Focus on what improved, not what could be better.

Respond in JSON format:
{
    "improvements": [
        {"aspect": "Color Balance", "percent": 35, "description": "Colors now appear more natural and vibrant"},
        {"aspect": "Clarity", "percent": 25, "description": "Image is noticeably sharper with better detail"}
    ],
    "overall_percent": 40,
    "summary": "A celebratory summary of the transformation (2-3 sentences, be positive!)"
}"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "ORIGINAL image:"
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
                            "text": "PROCESSED image:"
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
                
                return AIComparisonResult(
                    improvements=data.get("improvements", []),
                    overall_improvement=float(data.get("overall_percent", 0)),
                    summary=data.get("summary", "Processing complete!"),
                    success=True
                )
                
            except json.JSONDecodeError:
                return AIComparisonResult(
                    improvements=[],
                    overall_improvement=0,
                    summary=raw_text,
                    success=True,
                    error_message="Could not parse response"
                )
                
        except Exception as e:
            return AIComparisonResult(
                improvements=[],
                overall_improvement=0,
                summary="Analysis failed",
                success=False,
                error_message=str(e)
            )
    
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
        Specifically optimized for polaroid photos where the thick white border
        should always be at the BOTTOM.
        
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
            
            prompt = """Analyze this POLAROID photo to determine if it needs rotation.

POLAROID RULE: The thick white border (signature strip) must be at the BOTTOM.

DIRECTIONS IN THE IMAGE:
- TOP = upper edge of the image as you're viewing it now
- BOTTOM = lower edge of the image as you're viewing it now  
- LEFT = left side of the image as you're viewing it now
- RIGHT = right side of the image as you're viewing it now

YOUR TASK:
1. Find the polaroid frame's four white borders
2. One border is MUCH THICKER (2-3x wider) than the other three - this is the signature strip
3. Tell me which edge of the IMAGE (top/bottom/left/right) this thick border is on RIGHT NOW
4. Calculate rotation to move it to the bottom

ROTATION RULES:
- Thick border currently at BOTTOM → 0° (correct, no rotation)
- Thick border currently at TOP → 180° rotation
- Thick border currently on LEFT side → 270° clockwise
- Thick border currently on RIGHT side → 90° clockwise

VERIFICATION: After your proposed rotation, people should appear upright (not upside down or sideways).

JSON response only:
{
    "thick_border_location": "top/bottom/left/right",
    "border_analysis": "Which edge has the thick border and why",
    "people_check": "upright/upside_down/sideways_left/sideways_right/no_people",
    "needs_rotation": true/false,
    "rotation_degrees": 0/90/180/270,
    "confidence": "high/medium/low",
    "reasoning": "Your complete analysis"
}"""

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
            
            raw_text = response.content[0].text
            
            try:
                if "```json" in raw_text:
                    json_str = raw_text.split("```json")[1].split("```")[0]
                elif "```" in raw_text:
                    json_str = raw_text.split("```")[1].split("```")[0]
                else:
                    json_str = raw_text
                
                data = json.loads(json_str.strip())
                
                # Build detailed description from analysis
                thick_loc = data.get("thick_border_location", "unknown")
                border_analysis = data.get("border_analysis", "")
                people_check = data.get("people_check", "")
                reasoning = data.get("reasoning", "")
                
                description = f"Thick border at {thick_loc}. {border_analysis}"
                if people_check and people_check != "no_people":
                    description += f" People appear {people_check}."
                if reasoning:
                    description += f" {reasoning}"
                
                return OrientationResult(
                    needs_rotation=data.get("needs_rotation", False),
                    rotation_degrees=data.get("rotation_degrees", 0),
                    confidence=data.get("confidence", "low"),
                    description=description,
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

