import argparse
import json
from pathlib import Path

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import DynamicItemDataset
from torch.nn.functional import pad
from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor

from prepare_manifests import MANIFESTS, WAV_DIR
from train import HifiGanBrain

HERE = Path(__file__).resolve().parent
LAB_DIR = HERE.parent
RECIPE = HERE / "train.yaml"


def make_dataset(manifest, ap, mel_from_audio, segment_size):
    @sb.utils.data_pipeline.takes("wav", "segment")
    @sb.utils.data_pipeline.provides("mel", "sig")
    def read_pair(wav, segment):
        signal = torch.as_tensor(ap.load_wav(wav), dtype=torch.float32)

        if segment:
            max_start = max(0, signal.numel() - segment_size)
            start = torch.randint(max_start + 1, ()).item()
            signal = signal[start:start + segment_size]
            signal = pad(signal, (0, segment_size - signal.numel()))

        return mel_from_audio(signal), signal.unsqueeze(0)

    return DynamicItemDataset.from_json(
        json_path=str(manifest),
        dynamic_items=[read_pair],
        output_keys=["id", "mel", "sig"],
    )


def main():
    parser = argparse.ArgumentParser(description="Дообучение HiFi-GAN на RUSLAN")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs" / "full")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Для этого запуска нужна доступная GPU.")

    torch.backends.cudnn.enabled = False

    train_json = MANIFESTS / "train.json"
    valid_json = MANIFESTS / "valid.json"
    if not train_json.is_file() or not valid_json.is_file():
        raise SystemExit("Сначала prepare_manifests.py.")

    run_dir = args.run_dir.resolve()
    overrides = {
        "data_folder": str(WAV_DIR),
        "output_folder": str(run_dir),
        "train_json": str(train_json),
        "valid_json": str(valid_json),
        "skip_prep": True,
        "sample_rate": 22050,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 2e-5,
        "keep_checkpoint_interval": None,
        "mel_spectogram": None,
        "train_dataloader_opts": {"num_workers": 0},
        "valid_dataloader_opts": {"num_workers": 0},
    }

    with RECIPE.open(encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f, overrides)

    sb.create_experiment_directory(
        experiment_directory=str(run_dir),
        hyperparams_to_save=str(RECIPE),
        overrides=overrides,
    )

    config_path = LAB_DIR.parent / "lab4_acoustic" / "saved_runs" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ap = AudioProcessor.init_from_config(BaseAudioConfig(**config["audio"]))

    def mel_from_audio(audio):
        waveform = audio.detach().cpu().numpy().reshape(-1)
        return torch.as_tensor(ap.melspectrogram(waveform), dtype=torch.float32)

    hparams["mel_spectogram"] = mel_from_audio

    weights_path = LAB_DIR / "pretrained_models" / "hifigan" / "generator.ckpt"
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    hparams["generator"].load_state_dict(weights, strict=True)
    print(f"Загружены исходные веса: {weights_path}", flush=True)

    train_data = make_dataset(train_json, ap, mel_from_audio, hparams["segment_size"])
    valid_data = make_dataset(valid_json, ap, mel_from_audio, hparams["segment_size"])

    brain = HifiGanBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
            hparams["sch_class_generator"],
            hparams["sch_class_discriminator"],
        ],
        hparams=hparams,
        run_opts={"device": "cuda:0", "ckpt_interval_minutes": 10},
        checkpointer=hparams["checkpointer"],
    )

    print(f"RUSLAN: train={len(train_data)}, valid={len(valid_data)}", flush=True)
    brain.fit(
        brain.hparams.epoch_counter,
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs={**hparams["train_dataloader_opts"], "shuffle": True},
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    checkpoint = brain.checkpointer.find_checkpoint(min_key="G_l1_spec_loss")
    if checkpoint is None:
        raise SystemExit("Обучение завершилось без сохранённого чекпоинта.")
    print(f"Чекпоинт генератора: {checkpoint.paramfiles['generator']}")


if __name__ == "__main__":
    main()
