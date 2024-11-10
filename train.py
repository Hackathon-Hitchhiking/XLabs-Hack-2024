import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from vits import VITSModel, HiFiGAN
import librosa
import numpy as np

# Configuration
BERT_MODEL = 'bert-base-multilingual-cased'
SAMPLE_RATE = 22050
N_MELS = 80
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0002

# Initialize BERT model for style embeddings
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
bert_model = BertModel.from_pretrained(BERT_MODEL)

# Initialize VITS model and HiFi-GAN vocoder
model = VITSModel(config_path="config.yaml")
vocoder = HiFiGAN(config_path="config_hifigan.yaml")

# Helper function for generating mel-spectrograms
def save_mel_spectrogram(audio_path, mel_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    np.save(mel_path, mel_db)

# BERT embedding generation for style descriptions
def get_style_embedding(description):
    inputs = tokenizer(description, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Function to preprocess lyrics with punctuation handling
def preprocess_lyrics_with_punctuation(lyrics):
    phonemes = convert_to_phonemes(lyrics)  # Convert text to phonemes
    phonemes_with_pauses = []
    for phoneme in phonemes:
        phonemes_with_pauses.append(phoneme)
        if phoneme in [",", ".", ";", ":", "!"]:  # Add pause for punctuation
            phonemes_with_pauses.append("PAUSE")
    return phonemes_with_pauses

# Dataset loading and preprocessing
class SVSDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        self.data = []  # Load metadata with lyrics, audio paths, etc.
        with open(metadata_path, 'r') as f:
            for line in f:
                song_id, lyrics, phoneme_path, style, audio_path, mel_path = line.strip().split(',')
                style_embedding = get_style_embedding(style)
                self.data.append({
                    "lyrics": lyrics,
                    "phoneme_path": phoneme_path,
                    "style_embedding": style_embedding,
                    "audio_path": audio_path,
                    "mel_path": mel_path
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        lyrics = preprocess_lyrics_with_punctuation(item["lyrics"])
        mel_spectrogram = np.load(item["mel_path"])  # Precomputed mel-spectrogram
        style_embedding = item["style_embedding"]
        return lyrics, mel_spectrogram, style_embedding

# Collate function for dataloader
def collate_fn(batch):
    lyrics, mel_spectrograms, style_embeddings = zip(*batch)
    style_embeddings = torch.stack(style_embeddings)
    return lyrics, mel_spectrograms, style_embeddings

# Load dataset
dataset = SVSDataset(metadata_path="dataset_root/metadata.csv")
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

# Training pipeline
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    for lyrics, mel_spectrograms, style_embeddings in train_loader:
        lyrics_input = [preprocess_lyrics_with_punctuation(lyric) for lyric in lyrics]  # Convert to phonemes
        
        # Convert to tensor and process each segment in long lyrics
        mel_pred = torch.zeros_like(mel_spectrograms)
        for i, (lyric, style) in enumerate(zip(lyrics_input, style_embeddings)):
            mel_pred[i] = model(lyric, style)
        
        # Loss and optimization
        loss = torch.nn.functional.l1_loss(mel_pred, torch.tensor(mel_spectrograms))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")

# Save the model checkpoint
torch.save(model.state_dict(), "vits_style_model.pt")