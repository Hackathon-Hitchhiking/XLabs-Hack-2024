import sys

sys.path.insert(0, "Singer")
sys.path.insert(0, "research")

import torch
import Singer
import research
import fairseq
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

def generate_voice(text, model_path, data_cfg_path):
    """
    Generates speech from input text using a pre-trained TTS model.

    Args:
        text (str): The input text to be converted to speech.
        model_path (str): Path to the pre-trained TTS model checkpoint (.pt file).
        data_cfg_path (str): Path to the data configuration file (data.yaml).

    Returns:
        tuple: A tuple containing the waveform tensor and the sample rate.
    """
    # Load the pre-trained TTS model and task
    models, cfg, task = load_model_ensemble_and_task(
        [model_path],
        arg_overrides={"data_config": data_cfg_path}
    )
    model = models[0]
    print("cfg:", cfg)
    inference = TTSHubInterface(cfg, task, model)
    # Build the generator for inference
    generator = task.build_generator([model], cfg)

    # Prepare the input text
    text_inputs = text.strip()
    inputs = inference.get_model_input(task, text_inputs)

    # Generate the speech waveform
    with torch.no_grad():
        waveform, sample_rate = inference.get_prediction(task, model, generator, inputs)

    return waveform, sample_rate

# Example usage:
# Replace 'path/to/tts_model.pt' and 'path/to/data.yaml' with your actual paths.
text_to_speak = "Hello, this is a test."
waveform, sr = generate_voice(text_to_speak, '/Users/rebelraider/.cache/huggingface/hub/models--Cyanbox--Prompt-Singer/snapshots/4a9ad215081865e168df26ea4a54cc78e87378a6/prompt-singer-flant5-large-finetuned/checkpoint_last.pt', 'huita.yaml')

# You can then save the waveform to an audio file using a library like librosa or soundfile.
# For example:
import soundfile as sf
sf.write('output.wav', waveform.numpy(), sr)