"""this module handles speech to text"""
from dataclasses import dataclass
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

@dataclass
class WhisperInference:
    "The Inference Module will be used to pass text to the whisper model for inference"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id: str = "openai/whisper-medium"
    model = None
    pipe = None

    def __post_init__(self):
        self.setup()

    def setup(self):
        """setup for whisper"""
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            generate_kwargs={"language": "en"},
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
