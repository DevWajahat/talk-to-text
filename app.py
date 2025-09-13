import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import wave
import io
import numpy as np
import resampy
from tqdm import tqdm
from fpdf import FPDF
import time
import requests
import re

# --- Setup for PDF Generation ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'TalkToText Pro: Meeting Notes', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def parse_markdown_to_pdf(pdf_obj, markdown_text):
    """
    Parses a limited set of Markdown characters and applies FPDF styling.
    Supported: ## (H2), **bold**, *italic*, - (bullet points), and lists (*, -).
    """
    lines = markdown_text.split('\n')
    for line in lines:
        stripped_line = line.strip()
        
        # Heading 2 (##)
        if stripped_line.startswith('## '):
            pdf_obj.set_font('Arial', 'B', 14)
            pdf_obj.multi_cell(0, 10, stripped_line[3:])
            pdf_obj.ln(2)
            pdf_obj.set_font('Arial', '', 10) # Reset font to normal
            
        # Bullet Points (* or -)
        elif stripped_line.startswith(('* ', '- ')):
            pdf_obj.set_font('Arial', '', 10)
            pdf_obj.cell(10, 5, '•')
            pdf_obj.multi_cell(0, 5, stripped_line[2:])
            pdf_obj.ln(1)
            
        else:
            # Handle inline formatting (bold and italic)
            parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    pdf_obj.set_font('Arial', 'B', 10)
                    pdf_obj.write(5, part[2:-2])
                    pdf_obj.set_font('Arial', '', 10)
                elif part.startswith('*') and part.endswith('*'):
                    pdf_obj.set_font('Arial', 'I', 10)
                    pdf_obj.write(5, part[1:-1])
                    pdf_obj.set_font('Arial', '', 10)
                else:
                    pdf_obj.write(5, part)
            pdf_obj.ln(5)

def create_pdf(filename, transcription, summary):
    pdf = PDF()
    pdf.add_page()
    
    # Add title and filename
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f'Meeting: {filename}', ln=1)
    
    # Add transcription section
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Full Transcription', ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, transcription)
    
    pdf.ln(10)
    
    # Add structured notes section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Structured Notes & Summary', ln=1)
    pdf.set_font("Arial", size=10)
    
    # Use the new parsing function for the summary
    parse_markdown_to_pdf(pdf, summary)

    pdf_output_path = f"meeting_notes_{int(time.time())}.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path

# Set a professional Streamlit page configuration
st.set_page_config(
    page_title="TalkToText Pro: Open-Source Edition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Streamlit Session State ---
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# --- Initialize Hugging Face Models ---
@st.cache_resource
def load_whisper_model():
    st.write("Loading Whisper model... This may take a moment. ☕")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    st.success("Successfully loaded pre-trained Whisper model!")
    return processor, model

# Function to call the Gemini API
def get_gemini_response(prompt_text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': api_key
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Summarize this content \n" + prompt_text 
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# --- UI Layout ---
st.title("TalkToText Pro: AI-Powered Meeting Notes (Open-Source)")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload Audio File 📁")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav'],
        help="Only uncompressed WAV files are supported. They will be automatically resampled to 16,000 Hz."
    )
    
    st.subheader("Future Features")
    st.info("Input a link from Google Meet or Zoom (Coming Soon!)")

with col2:
    st.header("Meeting Transcription & Summary 📝")
    
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        
        if uploaded_file.name != st.session_state.uploaded_file_name:
            st.session_state.transcription = ""
            st.session_state.summary = ""
            st.session_state.uploaded_file_name = uploaded_file.name

        if not st.session_state.transcription:
            try:
                processor, whisper_model = load_whisper_model()
                
                audio_bytes_io = io.BytesIO(uploaded_file.getvalue())
                
                with wave.open(audio_bytes_io, 'rb') as wave_file:
                    n_channels = wave_file.getnchannels()
                    sample_rate = wave_file.getframerate()
                    n_frames = wave_file.getnframes()
                    audio_frames = wave_file.readframes(n_frames)
                
                audio_data = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32) / 32768.0

                if n_channels > 1:
                    st.info("Stereo audio detected. Converting to mono for transcription...")
                    stereo_data = audio_data.reshape(-1, n_channels)
                    mono_audio = np.mean(stereo_data, axis=1)
                else:
                    mono_audio = audio_data

                if sample_rate != 16000:
                    st.info(f"Resampling audio from {sample_rate} Hz to 16,000 Hz...")
                    resampled_audio = resampy.resample(mono_audio, sr_orig=sample_rate, sr_new=16000)
                else:
                    resampled_audio = mono_audio

                st.subheader("Raw Transcription")
                
                total_duration_s = len(resampled_audio) / 16000
                chunk_length_s = 30
                overlap_s = 5
                total_chunks = int(np.ceil(total_duration_s / chunk_length_s))
                full_transcription = []
                
                progress_bar = st.progress(0, text="Transcribing audio...")
                
                for i in tqdm(range(total_chunks)):
                    start_s = i * (chunk_length_s - overlap_s)
                    end_s = start_s + chunk_length_s
                    start_idx = int(start_s * 16000)
                    end_idx = int(end_s * 16000)
                    audio_chunk = resampled_audio[start_idx:end_idx]
                    if len(audio_chunk) == 0:
                        continue

                    input_features = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features
                    predicted_ids = whisper_model.generate(input_features)
                    transcription_chunk = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    full_transcription.append(transcription_chunk)
                    progress_bar.progress((i + 1) / total_chunks, text=f"Processing chunk {i+1} of {total_chunks}")
                
                final_transcript = " ".join(full_transcription)
                st.session_state.transcription = final_transcript
                
                st.subheader("Structured Meeting Notes")
                with st.spinner("Generating structured notes with Gemini..."):
                    gemini_prompt = f"""
                    You are a professional meeting assistant. Your task is to analyze the following audio transcription of a meeting and generate comprehensive, structured meeting notes and also translate transcription in english if there in any other language.

                    The output should be a single markdown document that includes:
                    
                    ## Executive Summary
                    A 3-4 sentence summary of the key discussion points and outcomes.

                    ## Key Discussion Points
                    * A bullet-point list summarizing the main topics covered.
                    
                    Here is the meeting transcription:
                    
                    {final_transcript}
                    """
                    
                    gemini_api_key = st.secrets["GEMINI_API_KEY"]
                    summary = get_gemini_response(gemini_prompt, gemini_api_key)
                    
                    if summary:
                        st.session_state.summary = summary
                    else:
                        st.session_state.summary = "Summary generation failed. Please check the API key and try again."
            
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.info("Please ensure the WAV file is uncompressed and valid.")
                st.session_state.transcription = ""
                st.session_state.summary = ""

        if st.session_state.transcription:
            st.text_area("Full Transcript", st.session_state.transcription, height=200)
            
            st.subheader("Structured Meeting Notes")
            st.markdown(st.session_state.summary)

            st.markdown("---")
            st.subheader("Export to PDF 📄")
            if st.button("Generate PDF"):
                if st.session_state.transcription and st.session_state.summary:
                    pdf_path = create_pdf(st.session_state.uploaded_file_name, st.session_state.transcription, st.session_state.summary)
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_file,
                            file_name=os.path.basename(pdf_path),
                            mime="application/octet-stream"
                        )
                    st.success("PDF report generated and ready for download!")
                else:
                    st.error("Please process the audio first to generate the transcription and summary.")

    else:
        st.info("Please upload a WAV file to get started.")

st.markdown("---")
st.markdown("Powered by Streamlit and open-source models from Hugging Face and Gemini.")