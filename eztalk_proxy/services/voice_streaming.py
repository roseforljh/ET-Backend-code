import logging
import base64
import orjson
import re
from typing import AsyncGenerator, Dict, Any
from .voice import google_handler, openai_handler, minimax_handler, siliconflow_handler

logger = logging.getLogger("EzTalkProxy.Services.VoiceStreaming")

class VoiceStreamProcessor:
    def __init__(
        self,
        stt_config: Dict[str, Any],
        chat_config: Dict[str, Any],
        tts_config: Dict[str, Any],
    ):
        self.stt_config = stt_config
        self.chat_config = chat_config
        self.tts_config = tts_config
        self.sentence_buffer = ""
        self.full_assistant_text = ""
        
        # Sentence splitters: ., ?, !, 。, ？, ！, and newlines
        self.sentence_endings = re.compile(r'[.?!。？！\n]+')

    async def process(self, audio_bytes: bytes, mime_type: str) -> AsyncGenerator[bytes, None]:
        """
        Execute the full STT -> Chat Stream -> TTS Stream pipeline.
        Yields NDJSON bytes.
        """
        # 1. STT
        user_text = await self._run_stt(audio_bytes, mime_type)
        if not user_text:
            yield self._format_event("error", {"message": "STT failed or empty result"})
            return

        # Yield initial meta event with user text
        yield self._format_event("meta", {
            "user_text": user_text,
            "assistant_text": ""
        })

        # 2. Chat Stream
        try:
            async for token in self._run_chat_stream(user_text):
                self.sentence_buffer += token
                self.full_assistant_text += token

                # Check for sentence completion
                if self._is_sentence_complete(self.sentence_buffer):
                    # Process the complete sentence(s)
                    sentences = self._extract_sentences()
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                            
                        # Yield updated text
                        yield self._format_event("meta", {
                            "user_text": user_text,
                            "assistant_text": self.full_assistant_text
                        })
                        
                        # 3. TTS Stream (Stop-and-Go: Await TTS for this sentence)
                        async for audio_chunk in self._run_tts_stream(sentence):
                            if audio_chunk:
                                yield self._format_event("audio", {
                                    "data": base64.b64encode(audio_chunk).decode('utf-8')
                                })

            # Flush remaining buffer
            if self.sentence_buffer.strip():
                yield self._format_event("meta", {
                    "user_text": user_text,
                    "assistant_text": self.full_assistant_text
                })
                async for audio_chunk in self._run_tts_stream(self.sentence_buffer):
                    if audio_chunk:
                        yield self._format_event("audio", {
                            "data": base64.b64encode(audio_chunk).decode('utf-8')
                        })

        except Exception as e:
            logger.exception("Streaming pipeline error")
            yield self._format_event("error", {"message": str(e)})

    async def _run_stt(self, audio_bytes: bytes, mime_type: str) -> str:
        platform = self.stt_config.get("platform", "Google")
        try:
            if platform == "OpenAI":
                return await openai_handler.process_stt(
                    audio_bytes=audio_bytes,
                    api_key=self.stt_config["api_key"],
                    api_url=self.stt_config.get("api_url"),
                    model=self.stt_config["model"]
                )
            elif platform == "SiliconFlow":
                return await siliconflow_handler.process_stt(
                    audio_bytes=audio_bytes,
                    api_key=self.stt_config["api_key"],
                    api_url=self.stt_config["api_url"],
                    model=self.stt_config["model"],
                    mime_type=mime_type
                )
            else: # Google
                return google_handler.process_stt(
                    audio_bytes=audio_bytes,
                    mime_type=mime_type,
                    api_key=self.stt_config["api_key"],
                    model=self.stt_config["model"],
                    api_url=self.stt_config.get("api_url")
                )
        except Exception as e:
            logger.error(f"STT failed: {e}")
            return ""

    async def _run_chat_stream(self, user_text: str):
        platform = self.chat_config.get("platform", "Google")
        
        # Construct system prompt
        system_prompt = self.chat_config.get("system_prompt", "")
        
        # SiliconFlow and other OpenAI-compatible platforms
        if platform == "OpenAI" or platform == "SiliconFlow":
            async for chunk in openai_handler.process_chat_stream(
                user_text=user_text,
                chat_history=self.chat_config.get("history", []),
                system_prompt=system_prompt,
                api_key=self.chat_config["api_key"],
                api_url=self.chat_config.get("api_url"),
                model=self.chat_config["model"]
            ):
                yield chunk
        else: # Google
            for chunk in google_handler.process_chat_stream(
                user_text=user_text,
                chat_history=self.chat_config.get("history", []),
                system_prompt=system_prompt,
                api_key=self.chat_config["api_key"],
                model=self.chat_config["model"],
                api_url=self.chat_config.get("api_url")
            ):
                yield chunk

    async def _run_tts_stream(self, text: str):
        platform = self.tts_config.get("platform", "Gemini")
        voice_name = self.tts_config.get("voice_name", "Kore")
        
        try:
            if platform == "Minimax":
                async for chunk in minimax_handler.synthesize_minimax_t2a_stream(
                    text=text,
                    voice_id=voice_name,
                    api_url=self.tts_config["api_url"],
                    api_key=self.tts_config["api_key"]
                ):
                    yield chunk
            elif platform == "SiliconFlow":
                async for chunk in siliconflow_handler.process_tts_stream(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    api_url=self.tts_config["api_url"],
                    model=self.tts_config["model"],
                    voice=voice_name
                ):
                    yield chunk
            elif platform == "OpenAI":
                # OpenAI TTS is not truly streaming (returns full file), but we can chunk it
                wav_data = await openai_handler.process_tts(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    voice_name=voice_name,
                    model=self.tts_config["model"],
                    api_url=self.tts_config.get("api_url")
                )
                # Strip WAV header if possible, or just send as is (Client handles it)
                # For simplicity, we send the whole blob as one chunk
                yield wav_data
            else: # Gemini
                pcm_data = google_handler.process_tts(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    voice_name=voice_name,
                    model=self.tts_config["model"],
                    api_url=self.tts_config.get("api_url")
                )
                yield pcm_data
        except Exception as e:
            logger.error(f"TTS stream failed for chunk '{text[:20]}...': {e}")

    def _is_sentence_complete(self, text: str) -> bool:
        # Simple check for sentence ending punctuation
        return bool(self.sentence_endings.search(text))

    def _extract_sentences(self) -> list[str]:
        """
        Split buffer into sentences, keeping the remaining incomplete part in buffer.
        """
        # Find all sentence splits
        parts = self.sentence_endings.split(self.sentence_buffer)
        matches = self.sentence_endings.findall(self.sentence_buffer)
        
        sentences = []
        new_buffer = ""
        
        # Reconstruct sentences with their punctuation
        for i in range(len(parts)):
            part = parts[i]
            if i < len(matches):
                punctuation = matches[i]
                sentences.append(part + punctuation)
            else:
                # The last part is the incomplete buffer
                new_buffer = part
        
        self.sentence_buffer = new_buffer
        return sentences

    def _format_event(self, event_type: str, data: Dict[str, Any]) -> bytes:
        payload = {"type": event_type, **data}
        return orjson.dumps(payload) + b"\n"
