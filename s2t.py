# import whisper

# model = whisper.load_model("base")
# result = model.transcribe("datasets/ttsmaker-file-2024-8-21-23-29-40.mp3    ")
# print(result["text"])

from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", 
                       model="vinai/PhoWhisper-small")
output = transcriber("datasets/PVfzM7B6o3o.wav")["text"]


print(output)
