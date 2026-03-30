"""Full empathic audio response pipeline.

dialogue_history (Avamerg turn)
    → EmpathyGenerator (LLM CoT) → response text
    → TTSModule (edge-tts)        → response .wav (16 kHz mono)
    → [optional] Audio2Lip        → lip motion tensor for EDTalk generator

Usage (text + audio only, no EDTalk models needed):
    pipeline = AudioResponsePipeline.from_avamerg_meta(meta)
    result   = pipeline.run_turn(turn, output_dir="outputs/")
    # result.response_text  → str
    # result.audio_path     → str  (path to WAV)

Usage (full, with EDTalk lip synthesis):
    pipeline = AudioResponsePipeline.from_avamerg_meta(meta,
                   audio2lip_model_path="ckpts/Audio2Lip.pt",
                   device="cuda")
    result = pipeline.run_turn(turn, output_dir="outputs/")
    # result.lip_features   → torch.Tensor (T, 20)  or None
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from .empathy_generator import EmpathyGenerator
from .tts_module import TTSModule

# EDTalk root is one level up from this file's directory
_EDTALK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class EmpathyResult:
    """Output of one pipeline run."""

    # Text outputs (always populated)
    response_text:   str = ""
    speaker_emotion: str = ""
    event_scenario:  str = ""
    emotion_cause:   str = ""
    goal_to_response: str = ""

    # Audio output (populated after TTS)
    audio_path: Optional[str] = None

    # Lip motion features (populated only when Audio2Lip model is loaded)
    # Shape: (T, 20) float32 tensor
    lip_features: Optional[object] = None  # torch.Tensor | None


class AudioResponsePipeline:
    """End-to-end pipeline: Avamerg turn → empathic spoken listener response.

    Args:
        empathy_generator: Configured EmpathyGenerator instance.
        tts_module:        Configured TTSModule instance.
        audio2lip:         Optional pre-loaded Audio2Lip model (nn.Module).
                           When provided, lip features are also computed.
        device:            Torch device string (e.g. "cuda", "cpu").
    """

    def __init__(
        self,
        empathy_generator: EmpathyGenerator,
        tts_module: TTSModule,
        audio2lip=None,
        device: str = "cpu",
    ):
        self.empathy_generator = empathy_generator
        self.tts_module = tts_module
        self.audio2lip = audio2lip
        self.device = device

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_avamerg_meta(
        cls,
        meta: dict,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        audio2lip_model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> "AudioResponsePipeline":
        """Build a pipeline from an Avamerg top-level conversation dict.

        Args:
            meta:                  Top-level Avamerg dict (with listener_profile,
                                   speaker_profile, topic, turns, …).
            llm_model:             LLM model name for EmpathyGenerator.
            llm_base_url:          Optional OpenAI-compatible base URL.
            llm_api_key:           Optional API key.
            audio2lip_model_path:  Path to Audio2Lip.pt; skips lip synthesis if
                                   None.
            device:                Torch device ("cuda" / "cpu").
        """
        gen = EmpathyGenerator(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        tts = TTSModule(
            listener_profile=meta.get("listener_profile"),
        )
        audio2lip = None
        if audio2lip_model_path:
            audio2lip = cls._load_audio2lip(audio2lip_model_path, device)

        return cls(gen, tts, audio2lip=audio2lip, device=device)

    @staticmethod
    def _load_audio2lip(model_path: str, device: str):
        """Load the Audio2Lip model from EDTalk checkpoints."""
        import torch

        sys.path.insert(0, _EDTALK_ROOT)
        from networks.audio_encoder import Audio2Lip

        model = Audio2Lip().to(device)
        weight = torch.load(model_path, map_location=device)
        model.load_state_dict(weight["audio2lip"])
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run_turn(
        self,
        turn: dict,
        meta: dict,
        output_dir: str = "outputs",
        turn_id: Optional[str] = None,
    ) -> EmpathyResult:
        """Run the full pipeline for one Avamerg conversation turn.

        Args:
            turn:       One element of avamerg["turns"].
            meta:       Top-level Avamerg dict (speaker_profile, topic, …).
            output_dir: Directory where the output WAV is saved.
            turn_id:    Optional ID string used in the output filename.
                        Defaults to turn["turn_id"].

        Returns:
            EmpathyResult populated with text, audio path, and optionally
            lip features.
        """
        tid = turn_id or str(turn.get("turn_id", "0"))
        conv_id = meta.get("conversation_id", "unknown")

        # Stage 1 — empathic response text (LLM CoT)
        cot = self.empathy_generator.generate_from_avamerg_turn(turn, meta)
        result = EmpathyResult(
            response_text=cot.get("response", ""),
            speaker_emotion=cot.get("speaker_emotion", ""),
            event_scenario=cot.get("event_scenario", ""),
            emotion_cause=cot.get("emotion_cause", ""),
            goal_to_response=cot.get("goal_to_response", ""),
        )

        # Stage 2 — TTS: text → WAV
        os.makedirs(output_dir, exist_ok=True)
        wav_name = f"{conv_id}_turn{tid}_response.wav"
        wav_path = os.path.join(output_dir, wav_name)
        result.audio_path = self.tts_module.synthesise(result.response_text, wav_path)

        # Stage 3 — Audio2Lip: WAV → lip motion features (optional)
        if self.audio2lip is not None:
            result.lip_features = self._compute_lip_features(result.audio_path)

        return result

    def _compute_lip_features(self, wav_path: str):
        """Run Audio2Lip on a WAV file and return lip motion tensor (T, 20)."""
        import torch
        import numpy as np
        import sys

        sys.path.insert(0, _EDTALK_ROOT)
        import audio as audio_utils

        # Load and mel-spectrogram the WAV (matches demo_EDTalk_A.py logic)
        wav = audio_utils.load_wav(wav_path, 16000)
        sr, fps = 16000, 25
        bit_per_frames = sr / fps
        num_frames = int(len(wav) / bit_per_frames)
        audio_length = int(num_frames * bit_per_frames)

        # Crop / pad
        if len(wav) > audio_length:
            wav = wav[:audio_length]
        elif len(wav) < audio_length:
            wav = np.pad(wav, [0, audio_length - len(wav)], mode="constant")

        orig_mel = audio_utils.melspectrogram(wav).T  # (nframes, 80)
        syncnet_step = 16
        indiv_mels = []
        for i in range(num_frames):
            start_idx = int(80.0 * ((i - 2) / float(fps)))
            end_idx = start_idx + syncnet_step
            seq = [min(max(j, 0), orig_mel.shape[0] - 1) for j in range(start_idx, end_idx)]
            indiv_mels.append(orig_mel[seq, :].T)  # (80, 16)

        mel_tensor = torch.FloatTensor(np.array(indiv_mels))  # (T, 80, 16)
        mel_tensor = mel_tensor.unsqueeze(1).unsqueeze(0).to(self.device)  # (1, T, 1, 80, 16)

        bs, T = mel_tensor.shape[0], mel_tensor.shape[1]
        audiox = mel_tensor.view(-1, 1, 80, 16)  # (bs*T, 1, 80, 16)

        with torch.no_grad():
            lip_features = self.audio2lip(audiox, bs, T)[0]  # (T, 20)

        return lip_features
