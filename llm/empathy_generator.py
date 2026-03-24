"""Empathic response text generation using chain-of-empathy CoT prompting.

Supports any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, etc.).
Set OPENAI_API_KEY and optionally OPENAI_BASE_URL in your environment.
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Optional


_SYSTEM_PROMPT = """You are an empathic listener in a conversation.
Given the conversation history, reason step-by-step using chain-of-empathy:
1. Identify the speaker's emotion.
2. Describe the event/scenario causing that emotion.
3. Identify the root cause of the emotion.
4. Determine your goal in responding empathically.
5. Generate a single warm, natural empathic response sentence.

Respond ONLY in this JSON format:
{
  "speaker_emotion": "...",
  "event_scenario": "...",
  "emotion_cause": "...",
  "goal_to_response": "...",
  "response": "..."
}"""


def _format_dialogue(dialogue_history: list[dict]) -> str:
    lines = []
    for turn in dialogue_history:
        role = turn.get("role", "speaker").capitalize()
        utterance = turn.get("utterance", "")
        lines.append(f"{role}: {utterance}")
    return "\n".join(lines)


class EmpathyGenerator:
    """Generates empathic text responses using chain-of-empathy CoT.

    Args:
        model:    OpenAI-compatible model name (e.g. "gpt-4o", "llama3").
        base_url: API base URL. Defaults to OPENAI_BASE_URL env var or
                  "https://api.openai.com/v1".
        api_key:  API key. Defaults to OPENAI_API_KEY env var.
        temperature: Sampling temperature (default 0.7).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature

    def _call_llm(self, user_content: str) -> str:
        """POST to /chat/completions and return assistant content string."""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["choices"][0]["message"]["content"]

    def generate(
        self,
        dialogue_history: list[dict],
        speaker_profile: Optional[dict] = None,
        topic: Optional[str] = None,
    ) -> dict:
        """Generate an empathic response for the given dialogue history.

        Args:
            dialogue_history: List of {"role": "speaker"/"listener",
                              "utterance": "..."} dicts (from Avamerg JSON).
            speaker_profile:  Optional {"age", "gender", "timbre"} dict.
            topic:            Optional conversation topic string.

        Returns:
            dict with keys: speaker_emotion, event_scenario, emotion_cause,
            goal_to_response, response.
        """
        context_lines = []
        if topic:
            context_lines.append(f"Topic: {topic}")
        if speaker_profile:
            age = speaker_profile.get("age", "")
            gender = speaker_profile.get("gender", "")
            context_lines.append(f"Speaker: {age} {gender}")

        dialogue_text = _format_dialogue(dialogue_history)
        user_content = "\n".join(filter(None, context_lines + [dialogue_text]))

        raw = self._call_llm(user_content)

        # Parse JSON from response (handle markdown code fences if present)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError(f"LLM did not return valid JSON:\n{raw}")

        return result

    def generate_from_avamerg_turn(self, turn: dict, meta: dict) -> dict:
        """Convenience wrapper that unpacks an Avamerg JSON turn.

        Args:
            turn: One element of avamerg["turns"] with keys:
                  dialogue_history, context, chain_of_empathy, response.
            meta: Top-level Avamerg dict with speaker_profile, topic, etc.

        Returns:
            Same as generate(): CoT dict including "response" text.
        """
        return self.generate(
            dialogue_history=turn["dialogue_history"],
            speaker_profile=meta.get("speaker_profile"),
            topic=meta.get("topic"),
        )
