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
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def process_image(image_path):
    image = Image.open(image_path)
    print("Image format:", image.format)
    print("Image mode:", image.mode)
    print("Image size:", image.size)
    
    # Convert image to RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
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

def save_audio_to_file(audio_io, filename):
    with open(filename, "wb") as f:
        f.write(audio_io.read())
        audio_io.close()  # Close the BytesIO object after writing
    print("Audio file saved to:", filename)


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return redirect(request.url)
        photo = request.files['photo']
        if photo.filename == '':
            return redirect(request.url)
        if photo:
            print("GOT PHOTO")
            filename = photo.filename
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(photo_path)
            # Process the uploaded image
            caption = process_image(photo_path)
            print("CAPTION GENERATED")
            print(caption)
            # Generate audio from the caption
            audio_io = generate_audio(caption)
            print("AUDIO GENERATED")
            # Save audio to a temporary file
            audio_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{os.path.splitext(filename)[0]}.mp3')
            save_audio_to_file(audio_io, audio_temp_path)

            return render_template('result.html', caption=caption, audio_path=audio_temp_path)

    return redirect(url_for('index'))


        
if __name__ == '__main__':
    app.run(debug=True)
