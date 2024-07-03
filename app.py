import streamlit as st
import whisper
import numpy as np
import srt
import io
from datetime import timedelta
import pysrt
import time
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os
import shutil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to preprocess and transcribe the audio file
def transcribe_audio(audio_file_path, audio_file_language):
    model_name = 'medium.en'
    model = whisper.load_model(model_name)
    
    result = model.transcribe(audio_file_path, language='en', verbose=True)

    result_srt_list = []
    for i in result['segments']:
        result_srt_list.append(srt.Subtitle(index=i['id'], start=timedelta(seconds=i['start']), end=timedelta(seconds=i['end']), content=i['text'].strip()))

    composed_transcription = srt.compose(result_srt_list)

    return composed_transcription

# Function to prompt user for preferred language
def prompt_user_for_language():
    st.write("Please select your preferred language for translation:")
    language_options = {
        'Hindi': 'hi',
        'Tamil': 'ta',
        'Marathi': 'mr',
        'Malayalam': 'ml',
        'Urdu': 'ur',
        'French': 'fr',
        'Spanish': 'es',
        'German': 'de',
        'Italian': 'it',
        'Russian': 'ru',
        'Arabic': 'ar'
    }
    selected_language = st.selectbox("Select Language", [""] + list(language_options.keys()))
    if selected_language:
        language_code = language_options[selected_language]
        return language_code
    else:
        return None

# Function to translate subtitles for the selected language using Hugging Face models
def translate_subtitles(original_srt, target_language_code):
    if target_language_code == 'ta':
        model_name = 'Mr-Vicky-01/English-Tamil-Translator'
    else:
        model_name = f'Helsinki-NLP/opus-mt-en-{target_language_code}'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    translated_srt = pysrt.SubRipFile()
    for subtitle in original_srt:
        translated_text = language_translator(subtitle.text_without_tags, tokenizer, model)
        translated_subtitle = pysrt.SubRipItem(
            index=subtitle.index,
            start=subtitle.start,
            end=subtitle.end,
            text=translated_text,
        )
        translated_srt.append(translated_subtitle)

    return translated_srt

# Function to perform translation using Hugging Face models
def language_translator(text, tokenizer, model):
    tokenized = tokenizer([text], return_tensors='pt')
    translated = model.generate(**tokenized)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def preview_video(video_file_path):
    temp_video_path = "temp_video.mp4"
    # Copy the video file to a temporary location
    shutil.copyfile(video_file_path, temp_video_path)
    # Display the temporary video file using st.video
    st.video(temp_video_path)

# Main Streamlit app code
def main():
    st.title("YouTube Video Transcription and Translation")

    # Input box for YouTube link
    youtube_link = st.text_input("Paste the YouTube video link here:")

    if youtube_link:
        try:
            yt = YouTube(youtube_link)
            video_title = yt.title
            video_author = yt.author
            video_description = yt.description

            st.subheader("Video Info")
            st.write(f"Title: {video_title}")
            st.write(f"Author: {video_author}")
            st.write(f"Description: {video_description}")

            # Translation section
            target_language_code = prompt_user_for_language()

            if target_language_code is not None:
                with st.spinner("Downloading video and audio..."):
                    # Download the YouTube video and audio
                    video_file_path = yt.streams.filter(progressive=True, file_extension='mp4').first().download(output_path=".", filename=f"{video_title}.mp4")
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    audio_file_path = audio_stream.download(output_path=".", filename="youtube_audio")
                    st.write("Video and Audio Downloaded.")

                with st.spinner("Transcription process has begun..."):
                    # Transcribe the audio file
                    transcription_srt_content = transcribe_audio(audio_file_path, 'en')

                # Display the transcribed subtitles in a scrollable area
                st.subheader("Transcribed Subtitles")
                transcription_container = st.empty()
                transcription_container.text_area("Transcribed Subtitles", transcription_srt_content, height=200)

                # Save the transcription as an SRT file
                with open("transcription.srt", "w") as f:
                    f.write(transcription_srt_content)
                st.write("Transcription saved as 'transcription.srt'.")

                # Preview the downloaded video
                st.subheader("Preview Video")
                preview_video(video_file_path)

                with st.spinner("Translation process has begun..."):
                    original_srt = pysrt.open("transcription.srt")
                    translated_srt = translate_subtitles(original_srt, target_language_code)

                    # Save the translated subtitles to an SRT file
                    translated_srt_file_path = f'translated_subtitle_{target_language_code}.srt'
                    translated_srt.save(translated_srt_file_path)

                # Display the translated subtitles content in a scrollable area
                st.subheader("Translated Subtitles")
                translation_container = st.empty()
                with open(translated_srt_file_path, 'r') as translated_file:
                    translated_srt_content = translated_file.read()
                translation_container.text_area("Translated Subtitles", translated_srt_content, height=200)

                # Display a download link for the translated subtitles file
                st.subheader("Download Translated Subtitles")
                download_button = st.download_button(
                    label="Download Translated Subtitles",
                    data=translated_srt_content,
                    file_name=translated_srt_file_path,
                    mime="text/plain",
                )
                if download_button:
                    st.write("Translated subtitles downloaded.")
                else:
                    st.write("Click the 'Translate' button to start the translation process.")

                # Download button for video and SRT
                st.subheader("Download the Video")
                download_button = st.download_button(
                    label = "Download Video and SRT",
                    data=open(video_file_path, "rb").read(),
                    file_name=f"{video_title}.mp4",
                    mime="video/mp4",
                )
                if download_button:
                    with open("transcription.srt", "rb") as f:
                        srt_data = f.read()
                    st.download_button(
                        label="Download SRT",
                        data=srt_data,
                        file_name="transcription.srt",
                        mime="text/plain",
                    )

        except Exception as e:
            st.error(f"Error fetching video info: {e}")
    else:
        st.warning("Please paste a YouTube video link.")

if __name__ == "__main__":
    main()