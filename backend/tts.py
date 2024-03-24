"""This module is used to convert text to speech using the eleven labs api."""

from dataclasses import dataclass
import requests


@dataclass
class TtsResponse:
    """this object will be used to pass text to eleven labs and
    an audio file with the speech will be written to the fs"""

    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/L1hzywmmQI2XuvL5TMvY"
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

    def _save_audio(self, file_name):
        with open(file_name, "wb") as f:
            for chunk in self.response.iter_content(chunk_size=self.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    def create_data_packet(self, gpt_response: str, file_name: str):
        """create post request to eleven labs with response from gpt"""
        self.data.update({"text": gpt_response})
        self.response = requests.post(self.url, json=self.data, headers=self.headers, timeout=15)
        self._save_audio(file_name)
