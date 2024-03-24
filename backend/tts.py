"""This module is used to convert text to speech using the eleven labs api."""

from dataclasses import dataclass
import requests
import logging

logger = logging.getLogger(__name__)

@dataclass
class TtsResponse:
    """this object will be used to pass text to eleven labs and
    an audio file with the speech will be written to the fs"""

    # voice for shankar is an old french guy, kevin is a young american guy
    voice_label = {"shankar": "VwQZoIAyyglWE7jHRNDB", "kevin": "L1hzywmmQI2XuvL5TMvY"}
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "fdc44465af51d7867af8c28382701771",
    }

    data = {
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
    }
    response = None

    def __init__(self, voice_style: str):
        self.eleven_voice_style = voice_style
                         
    def _save_audio(self, file_name):
        with open(file_name, "wb") as f:
            for chunk in self.response.iter_content(chunk_size=self.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    def create_data_packet(self, gpt_response: str, file_name: str):
        import time
        """create post request to eleven labs with response from gpt"""
        self.data.update({"text": gpt_response})
        self.response = requests.post(self.url+self.voice_label[self.eleven_voice_style], json=self.data, headers=self.headers, timeout=15)
        start_time = time.perf_counter()
        self._save_audio(file_name)
        end_time = time.perf_counter()
        logger.info(f"Time taken to save audio: {end_time-start_time}")
