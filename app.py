from flask import Flask, render_template, request, redirect, url_for
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO

import os
import requests
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def process_image(image_path):
    image = Image.open(image_path)
    img = image_processor(image, return_tensors="pt").to(device)
    output = model.generate(**img)
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption

def generate_audio(caption):
    tts = gTTS(text=caption, lang='en')
    audio_io = BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return redirect(request.url)
        photo = request.files['photo']
        if photo.filename == '':
            return redirect(request.url)
        if photo:
            filename = photo.filename
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(photo_path)
            # Process the uploaded image
            caption = process_image(photo_path)
            # Generate audio from the caption
            audio_io = generate_audio(caption)
            # Convert audio format if needed
            audio = AudioSegment.from_file(audio_io, format="mp3")
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'caption_audio.mp3')
            audio.export(audio_path, format="mp3")
            return render_template('result.html', caption=caption, audio_path=audio_path)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
