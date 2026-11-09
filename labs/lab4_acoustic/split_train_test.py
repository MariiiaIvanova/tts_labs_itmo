from pathlib import Path

def split(src: Path, dst_dir: Path, stem: str) -> tuple[int, int]:
    train_lines, test_lines = [], []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("|")
            if len(cols) < 3:
                continue
            digits = cols[0].split("_")[0]
            if not digits.isdigit():
                continue
            if int(digits) % 5 == 0:
                test_lines.append(line)
            else:
                train_lines.append(line)
    (dst_dir / f"{stem}_train.csv").write_text("".join(train_lines), encoding="utf-8")
    (dst_dir / f"{stem}_test.csv").write_text("".join(test_lines), encoding="utf-8")
    return len(train_lines), len(test_lines)

def main() -> None:
    data = Path("../../data")
    print("metadata_phones (G2P):")
    tr, te = split(data / "metadata_phones.csv", data, "metadata_phones")
    print(f"  train {tr}  test {te}")
    print("metadata_phones_gt (GT):")
    tr, te = split(data / "metadata_phones_gt.csv", data, "metadata_phones_gt")
    print(f"  train {tr}  test {te}")
    print("\ncheck eval_sentences.csv ids are all in test:")
    ids = set()
    with (Path("data/eval_sentences.csv")).open(encoding="utf-8") as fh:
        for line in fh:
            ids.add(line.split("|")[0])
    print(f"  eval ids: {len(ids)}")
    gt_test = set()
    with (data / "metadata_phones_gt_test.csv").open(encoding="utf-8") as fh:
        for line in fh:
            gt_test.add(line.split("|")[0])
    missing = ids - gt_test
    print(f"  ids in eval but not in test split: {len(missing)}")
    if missing:
        print(f"  e.g. {sorted(missing)[:3]}")
    else:
        print("all 120 eval ids are in the test split")

if __name__ == "__main__":
    main()