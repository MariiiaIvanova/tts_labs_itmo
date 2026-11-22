import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
WAV_DIR = HERE.parents[2] / "data" / "RUSLAN" / "RUSLAN"
MANIFESTS = HERE / "manifests"


def main():
    train, held_out = [], []
    for path in sorted(WAV_DIR.glob("*_RUSLAN.wav")):
        number = path.stem.split("_", 1)[0]
        if number.isdigit():
            group = held_out if int(number) % 5 == 0 else train
            group.append(path)

    if not train or len(held_out) < 37:
        raise SystemExit(f"Недостаточно записей. Проверь путь: {WAV_DIR}")

    rng = random.Random(42)
    rng.shuffle(train)
    rng.shuffle(held_out)
    listen = held_out[:5]
    valid = held_out[5:37]

    MANIFESTS.mkdir(exist_ok=True)
    for name, paths, segment in [
        ("train", train, True),
        ("valid", valid, False),
        ("listen_test", listen, False),
    ]:
        rows = {p.stem: {"wav": str(p), "segment": segment} for p in paths}
        path = MANIFESTS / f"{name}.json"
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"{name}: {len(rows)} записей")


if __name__ == "__main__":
    main()
