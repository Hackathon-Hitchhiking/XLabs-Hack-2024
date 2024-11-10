import nemo.collections.asr as nemo_asr
import torch
import torchaudio

# Загружаем модель шумоподавления из NeMo
denoiser_model = nemo_asr.models.ENHANCE_Model.from_pretrained(model_name="nvidia/stft-denoiser")

# Загружаем файл вокала
vocal_waveform, sample_rate = torchaudio.load("only_vocal.wav")

# Приводим данные к нужному формату для модели
vocal_waveform = vocal_waveform.unsqueeze(0)  # Добавляем batch dimension

# Обработка вокала моделью NeMo для шумоподавления
with torch.no_grad():
    cleaned_waveform = denoiser_model(vocal_waveform)

# Сохраняем очищенный аудиофайл
torchaudio.save("cleaned_vocal_nemo.wav", cleaned_waveform.squeeze(0), sample_rate)