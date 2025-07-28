import os
import tempfile

import gradio as gr
import soundfile as sf
import torch
import whisper

from boson_multimodal.data_types import AudioContent, ChatMLSample, Message
from boson_multimodal.serve.serve_engine import HiggsAudioResponse, HiggsAudioServeEngine


# Model and tokenizer paths
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the HiggsAudio engine
serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

# Load Whisper model
whisper_model = whisper.load_model("base", device=device)

# Pre-loaded voices directory
voice_prompts_dir = "examples\\voice_prompts"
preloaded_voices = [f.split('.')[0] for f in os.listdir(voice_prompts_dir) if f.endswith('.wav')]


# Audio generation function
def generate_audio(scene_description, transcript, voice_type, ref_audio_dropdown, custom_audio_upload, temperature, seed, speaker0, speaker0_custom_audio_upload, speaker1, speaker1_custom_audio_upload):
    # Detect if transcript is multi-speaker
    is_multi_speaker = "[SPEAKER" in transcript
    if is_multi_speaker and voice_type == "Voice Clone":
        return "For multi-speaker transcripts, please use 'Smart Voice' or 'Multi-voice Clone'."
    if not is_multi_speaker and voice_type == "Multi-voice Clone":
        return "For 'Multi-voice Clone', your transcript must include speaker tags like [SPEAKER0] and [SPEAKER1]."

    # Construct system prompt
    system_prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
    messages = [Message(role="system", content=system_prompt)]
    ras_win_len = 7  # Default RAS window length

    # Handle voice type
    if voice_type == "Smart Voice":
        messages.append(Message(role="user", content=transcript))
    elif voice_type == "Voice Clone":
        ref_audio_path = ""
        ref_transcript = ""
        if ref_audio_dropdown == "Custom Upload":
            if custom_audio_upload is None:
                return "Please upload a custom audio file (WAV format)."
            ref_audio_path = custom_audio_upload
            result = whisper_model.transcribe(ref_audio_path)
            ref_transcript = result["text"]
        else:
            ref_audio_path = os.path.join(voice_prompts_dir, f"{ref_audio_dropdown}.wav")
            ref_transcript_path = os.path.join(voice_prompts_dir, f"{ref_audio_dropdown}.txt")
            if not os.path.exists(ref_transcript_path):
                return f"Reference transcript not found at {ref_transcript_path}"
            with open(ref_transcript_path, encoding="utf-8") as f:
                ref_transcript = f.read().strip()

        messages.append(Message(role="user", content=ref_transcript))
        messages.append(Message(role="assistant", content=[AudioContent(audio_url=ref_audio_path)]))
        messages.append(Message(role="user", content=transcript))

    elif voice_type == "Multi-voice Clone":
        if speaker0 == "None" or speaker1 == "None":
            return "Please select two speakers for multi-voice cloning."

        # Get the reference audio and transcript for each speaker
        ref_audio_path_0 = ""
        ref_transcript_0 = ""
        if speaker0 == "Custom Upload":
            if speaker0_custom_audio_upload is None:
                return "Please upload a custom audio file for Speaker 0 (WAV format)."
            ref_audio_path_0 = speaker0_custom_audio_upload
            result = whisper_model.transcribe(ref_audio_path_0)
            ref_transcript_0 = result["text"]
        else:
            ref_audio_path_0 = os.path.join(voice_prompts_dir, f"{speaker0}.wav")
            ref_transcript_path_0 = os.path.join(voice_prompts_dir, f"{speaker0}.txt")
            if not os.path.exists(ref_transcript_path_0):
                return f"Reference transcript not found for {speaker0}"
            with open(ref_transcript_path_0, encoding="utf-8") as f:
                ref_transcript_0 = f.read().strip()

        ref_audio_path_1 = ""
        ref_transcript_1 = ""
        if speaker1 == "Custom Upload":
            if speaker1_custom_audio_upload is None:
                return "Please upload a custom audio file for Speaker 1 (WAV format)."
            ref_audio_path_1 = speaker1_custom_audio_upload
            result = whisper_model.transcribe(ref_audio_path_1)
            ref_transcript_1 = result["text"]
        else:
            ref_audio_path_1 = os.path.join(voice_prompts_dir, f"{speaker1}.wav")
            ref_transcript_path_1 = os.path.join(voice_prompts_dir, f"{speaker1}.txt")
            if not os.path.exists(ref_transcript_path_1):
                return f"Reference transcript not found for {speaker1}"
            with open(ref_transcript_path_1, encoding="utf-8") as f:
                ref_transcript_1 = f.read().strip()

        # Construct the message sequence to teach the model the voices
        messages.extend([
            Message(role="user", content=f"[SPEAKER0] {ref_transcript_0}"),
            Message(role="assistant", content=[AudioContent(audio_url=ref_audio_path_0)]),
            Message(role="user", content=f"[SPEAKER1] {ref_transcript_1}"),
            Message(role="assistant", content=[AudioContent(audio_url=ref_audio_path_1)]),
            Message(role="user", content=transcript)
        ])

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(int(seed))

    # Generate audio
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        max_new_tokens=2048,
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        ras_win_len=ras_win_len,
        seed=seed
    )

    # Convert tensor -> numpy
    audio_data = output.audio.detach().cpu().numpy() if torch.is_tensor(output.audio) else output.audio

    # Save audio to a temp WAV file and return the path
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, audio_data, output.sampling_rate)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tmp_file.name  # Gradio Audio component will read this file path


# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as app:  # Dark theme replacement
    gr.Markdown(
        """
        # Get Going Fast: HiggsAudio Generation App
        Generate high-quality audio with HiggsAudio. Input a transcript, choose a voice type, and customize settings. 
        For multi-speaker dialogs, use tags like `[SPEAKER0]`, `[SPEAKER1]`, etc., with Smart Voice.

        [Listen to good music!](https://music.youtube.com/channel/UCY658vbL6S2zlRomNHoX54Q)
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            scene_description = gr.Textbox(
                label="Scene Description",
                value="Audio is recorded from a quiet room.",
                info="Describe the acoustic environment."
            )
            transcript = gr.Textbox(
                label="Transcript",
                lines=5,
                placeholder="Enter text here. For multi-speaker, use [SPEAKER0], [SPEAKER1], etc."
            )
            voice_type = gr.Radio(
                ["Smart Voice", "Voice Clone", "Multi-voice Clone"],
                label="Voice Type",
                value="Smart Voice",
                info="Smart Voice: Model selects voice. Voice Clone: Use a single reference audio. Multi-voice Clone: Use two reference audios for multi-speaker transcripts."
            )
            with gr.Group(visible=False) as voice_clone_group:
                ref_audio_dropdown = gr.Dropdown(
                    choices=["None"] + preloaded_voices + ["Custom Upload"],
                    label="Reference Audio",
                    value="None",
                    info="Select a pre-loaded voice or upload your own (used with Voice Clone)."
                )
                custom_audio_upload = gr.File(
                    label="Custom Reference Audio (Upload a WAV file if 'Custom Upload' is selected)",
                    file_types=[".wav"]
                )
            with gr.Group(visible=False) as multi_voice_clone_group:
                speaker0_dropdown = gr.Dropdown(
                    choices=["None"] + preloaded_voices + ["Custom Upload"],
                    label="Speaker 0",
                    value="belinda",
                    info="Select a pre-loaded voice or upload your own for [SPEAKER0]."
                )
                speaker0_custom_audio_upload = gr.File(
                    label="Custom Reference Audio for Speaker 0 (Upload a WAV file if 'Custom Upload' is selected)",
                    file_types=[".wav"],
                    visible=False
                )
                speaker1_dropdown = gr.Dropdown(
                    choices=["None"] + preloaded_voices + ["Custom Upload"],
                    label="Speaker 1",
                    value="broom_salesman",
                    info="Select a pre-loaded voice or upload your own for [SPEAKER1]."
                )
                speaker1_custom_audio_upload = gr.File(
                    label="Custom Reference Audio for Speaker 1 (Upload a WAV file if 'Custom Upload' is selected)",
                    file_types=[".wav"],
                    visible=False
                )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.1,
                value=0.3,
                label="Temperature",
                info="Controls randomness (0.1 = less random, 1.0 = more random)."
            )
            seed = gr.Number(
                label="Seed",
                value=12345,
                info="Set for reproducible results. Leave blank for random."
            )
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Generated Audio", interactive=False, type="filepath")

    # Connect the button to the generation function
    voice_type.change(fn=lambda value: (gr.update(visible=value == "Voice Clone"), gr.update(visible=value == "Multi-voice Clone")),
                        inputs=voice_type,
                        outputs=[voice_clone_group, multi_voice_clone_group])

    ref_audio_dropdown.change(fn=lambda value: gr.update(visible=value == "Custom Upload"),
                               inputs=ref_audio_dropdown,
                               outputs=custom_audio_upload)

    speaker0_dropdown.change(fn=lambda value: gr.update(visible=value == "Custom Upload"),
                              inputs=speaker0_dropdown,
                              outputs=speaker0_custom_audio_upload)

    speaker1_dropdown.change(fn=lambda value: gr.update(visible=value == "Custom Upload"),
                              inputs=speaker1_dropdown,
                              outputs=speaker1_custom_audio_upload)

    generate_btn.click(
        fn=generate_audio,
        inputs=[scene_description, transcript, voice_type, ref_audio_dropdown, custom_audio_upload, temperature, seed, speaker0_dropdown, speaker0_custom_audio_upload, speaker1_dropdown, speaker1_custom_audio_upload],
        outputs=output_audio
    )

# Launch the app
app.launch()
