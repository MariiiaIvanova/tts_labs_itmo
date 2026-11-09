"""Check that every phone in train metadata is in the inventory (NFC)."""
import unicodedata
from pathlib import Path

def norm(s):
    return unicodedata.normalize("NFC", s.strip())

inv_path = Path("../../common/phones_mfa.txt")
inv = {norm(l) for l in inv_path.read_text(encoding="utf-8").splitlines() if l.strip()}
print(f"inventory: {len(inv)} types from {inv_path}")

for f in ["../../data/metadata_phones_train.csv",
          "../../data/metadata_phones_gt_train.csv"]:
    seen, n = set(), 0
    with Path(f).open(encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("|")
            if len(cols) < 3:
                continue
            n += 1
            for ph in cols[2].split():
                seen.add(norm(ph))
    unknown = seen - inv
    print(f"\n{f}")
    print(f"  utts: {n}  distinct phones: {len(seen)}  unknown: {len(unknown)}")
    if unknown:
        for u in sorted(unknown)[:15]:
            print(f"  {u!r}  " + " ".join(f"U+{ord(c):04X}" for c in u))