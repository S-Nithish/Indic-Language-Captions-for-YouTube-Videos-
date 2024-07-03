# Indic-Language-Captions-for-YouTube-Videos-
Indic-Multilingual captioning solution by integrating Open AI's Whisper model with advanced language models (LLMs).

## Overview
* This Streamlit application allows users to input a YouTube video link, transcribe the audio, and translate the subtitles into a chosen language. It leverages Whisper for transcription and Hugging Face models for translation.
* The application also allows users to preview the video and download the translated subtitles.

## Features
* Transcribe Audio from YouTube Video: Extracts audio from a YouTube video and generates transcription.
* Translate Subtitles: Translates the generated subtitles into a user-selected language.
* Preview and Download: Provides a preview of the video and allows downloading the video along with the subtitles.

## Libraries Used
**streamlit:** For building the web application.
**whisper:** For transcribing audio.
**numpy:** For numerical operations.
**srt:** For handling SRT subtitles.
**pysrt:** For reading and writing SRT files.
**time, timedelta:** For handling time-related operations.
**pytube:** For downloading YouTube videos and audio.
**moviepy:** For handling video files.
**shutil:** For file operations.
**transformers (AutoTokenizer, AutoModelForSeq2SeqLM):** For translation models.
