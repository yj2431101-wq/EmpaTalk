"""LLM-based empathic audio response generation for EDTalk.

Pipeline:
    dialogue_history + chain_of_empathy
        → EmpathyGenerator (LLM)  → response text
        → TTSModule (edge-tts)    → response audio (.wav)
        → Audio2Lip               → listener lip motion
"""

from .empathy_generator import EmpathyGenerator
from .tts_module import TTSModule
from .audio_response_pipeline import AudioResponsePipeline

__all__ = ["EmpathyGenerator", "TTSModule", "AudioResponsePipeline"]
