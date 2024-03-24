# pylint: disable=W4901
"""prgram to setup conversational agents"""
import audioop
import base64
import json
import os
import logging
import wave
import random
import string
import requests


from flask import Flask, request
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.rest import Client
from dotenv import load_dotenv
from openai import OpenAI
import redis
from tts import TtsResponse
from aws_utils import upload_file_to_s3, create_presigned_url


logger = logging.getLogger(__name__)
load_dotenv()


client_gpt = OpenAI()
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client_twilio = Client(account_sid, auth_token)
redis_client = redis.Redis(host="localhost", port=6379, db=0)
mistral_api_key = os.environ["MISTRAL_API_KEY"]
MISTRAL_MODEL = "mistral-large-latest"


tts = TtsResponse()
app = Flask(__name__)
sock = Sock(app)
twilio_client = Client()
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
CL = "\x1b[0K"
BS = "\x08"


def generate_random_id(length=8):
    """Generate a random string of letters and digits"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def process_response(gpt_response, random_number):
    """this func will convert the gpt response to an audio file via eleven labs api
    store the file in s3 and pass the presigned url to twilio so it can play it"""
    calls = client_twilio.calls.list(status="in-progress")
    if len(calls) > 0:
        first_call = calls[0].sid
        tts.create_data_packet(gpt_response, f"eleven_{random_number}.mp3")
        if upload_file_to_s3(
            f"eleven_{random_number}.mp3",
            "construction-agent",
            f"eleven_{random_number}.mp3",
        ):
            # Generate presigned URL
            url = create_presigned_url(
                "construction-agent", f"eleven_{random_number}.mp3"
            )
            if url:
                print("Presigned URL: ", url)
            else:
                print("Could not generate presigned URL")
        else:
            print("Upload failed")
        client_twilio.calls(first_call).update(
            twiml=f"<Response><Play><![CDATA[{url}]]></Play><Pause length='100'/></Response>"
        )
    else:
        logger.info("no active call")


def conversation_chat_mistral(transcription, unique_id):
    """make gpt calls for the conversation"""
    context = [
        {
            "role": "system",
            "content": "You are a general purpose conversational"
            "agent being communicated with through the phone",
        }
    ]
    mistral_data = {"role": "system", "content": transcription}
    data_serialized = json.dumps(mistral_data)
    redis_client.rpush(unique_id, data_serialized)
    serialized_list = redis_client.lrange(unique_id, 0, -1)
    # Deserialize each JSON string back to a dictionary
    list_of_dicts = [json.loads(item) for item in serialized_list]

    url = "https://api.mistral.ai/v1/chat/completions"

    # Your API key (if required, replace 'Your_API_Key_Here' with your actual API key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mistral_api_key}",
    }

    support_context = [
        {"role": "system", "content": "The following content is purely for context"}
    ]
    payload = {
        "model": "mistral-large-latest",
        "messages": context
        + list_of_dicts[:-1]
        + [{"role": "user", "content": transcription}]
        + support_context,
        "temperature": 0.7,
        "top_p": 1,
        "max_tokens": 512,
        "stream": False,
        "safe_prompt": False,
        "random_seed": 1337,
    }
    response = requests.post(
        url, headers=headers, data=json.dumps(payload), timeout=100
    )
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        print("Response data:", data["choices"][0]["message"]["content"])

        mistral_response_data = {
            "role": "system",
            "content": data["choices"][0]["message"]["content"],
        }
        data_serialized = json.dumps(mistral_response_data)
        redis_client.rpush(unique_id, data_serialized)
        serialized_list = redis_client.lrange(unique_id, 0, -1)

        return data["choices"][0]["message"]["content"]

    return None


def conversation_chat(transcription, unique_id):
    """make gpt calls for the conversation"""
    data = {"role": "user", "content": transcription}
    data_serialized = json.dumps(data)
    redis_client.rpush(unique_id, data_serialized)
    serialized_list = redis_client.lrange(unique_id, 0, -1)
    # Deserialize each JSON string back to a dictionary
    list_of_dicts = [json.loads(item) for item in serialized_list]
    completion = client_gpt.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a receptionist at the Best Western in Chicago."
                "Your job is to assist guests by telling them information about Chicago"
                "and about the Hotel. The pool is open from monday to Friday"
                "from 9 am to 3 pm. The restaurant is open all 7 days from 9 am to 7 pm,"
                "the breakfast buffet is between 9 am and 11 am. Ask follow up questions"
                "in the conversation before answering questions.You do not know anything else,"
                "and don't assume anything about the hotel or the restaurant",
            },
            *list_of_dicts,
        ],
    )
    gpt_response_data = {
        "role": "assistant",
        "content": completion.choices[0].message.content,
    }
    gpt_data_serialized = json.dumps(gpt_response_data)
    redis_client.rpush(unique_id, gpt_data_serialized)
    return completion.choices[0].message.content


def transcribe_audio(audio_file_path: str) -> str:
    """convert speech to text"""
    with open(audio_file_path, "rb") as audio_file:
        # You can use audio_file here to read data etc.
        # For example, to read the entire content into a variable:

        transcript = client_gpt.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            prompt="ignore thank you and good bye",
        )

        if transcript.strip().lower() in ["thank you", "thank you."]:
            return ""

        return transcript


def process_audio_data(packet) -> bytes:
    """Process audio stream from twilio into bytes"""
    audio = base64.b64decode(packet["media"]["payload"])
    audio = audioop.ulaw2lin(audio, 2)
    audio = audioop.ratecv(audio, 2, 1, 8000, 8000, None)[0]
    return audio


@sock.route("/stream")
def stream(ws):
    """Receive and transcribe audio stream."""

    packet_count = 0
    random_audio_count = 0
    packed_data = []
    unique_id = generate_random_id(10)
    while True:
        message = ws.receive()
        packet = json.loads(message)

        if packet["event"] == "start":
            logger.info("Streaming is starting")
        elif packet["event"] == "stop":
            logger.info("\nStreaming has stopped")
        elif packet["event"] == "media":
            audio = process_audio_data(packet)
            packed_data.append(audio)

        if packet_count % 200 == 0 and packet_count != 0:
            frames = b"".join(packed_data)
            output = wave.open(f"output_{random_audio_count}.wav", "w")
            output.setparams((1, 2, 8000, 0, "NONE", "compressed"))
            output.writeframes(frames)
            output.close()
            packed_data.clear()
            result = transcribe_audio(f"output_{random_audio_count}.wav")

            if result:
                logger.info("%s%s ", CL, result)
                if len(result) > 10:
                    gpt_response = conversation_chat_mistral(result, unique_id)
                    process_response(gpt_response, random_audio_count)
            packet_count = 0
        packet_count += 1
        random_audio_count += 1


@app.route("/call", methods=["POST"])
def call():
    """Accept a phone call."""
    response = VoiceResponse()

    start = Start()
    if os.environ.get("DEVELOPMENT_ENV") == "local":
        print("i am here")
        print(f"wss://{request.host}/stream")
        start.stream(url=f"wss://{request.host}/stream")
    else:
        start.stream(url="wss://endpoint.elaralabs.com/stream")

    response.append(start)
    url = create_presigned_url("construction-agent", "intro-best-western.mp3")
    url_kevin_response = create_presigned_url(
        "construction-agent", "kevin_call_center_response.mp3"
    )
    url_shankar_response = create_presigned_url(
        "construction-agent", "shankar_call_center_response.mp3"
    )
    if url:
        print("Presigned URL: ", url)
    else:
        print("Could not generate presigned URL")

    caller_number = request.values.get("From", None)
    if caller_number == "+19089221772":
        response.play(url_kevin_response)
    elif caller_number == "+15182683957":
        response.play(url_shankar_response)
    else:
        response.play(url)
    response.pause(length=100)

    # this log won't work consistently because of the request.form. need to fix
    # logger.info("Incoming call from %s", request.form["From"])
    return str(response), 200, {"Content-Type": "text/xml"}


@app.route("/test", methods=["GET"])
def test():
    """Test service."""
    return "Service is working", 200


if __name__ == "__main__":
    from pyngrok import ngrok

    PORT = 5060
    if os.environ.get("DEVELOPMENT_ENV") == "local":
        public_url = ngrok.connect(PORT, bind_tls=True).public_url
        number = twilio_client.incoming_phone_numbers.list()[0]
        number.update(voice_url=public_url + "/call")
        logger.info(f"Waiting for calls on %s,{number.phone_number}")

    app.run(port=PORT)
