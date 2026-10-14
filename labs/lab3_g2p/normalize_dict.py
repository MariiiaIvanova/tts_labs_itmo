"""Strip the length mark ː from russian_mfa.dict and validate the phone inventory.

Replaces every phone `Xː` with `X`, maps the two rare stray phones (tsʲ, dzʲ) to their
closest target counterparts, then checks that all phones are present in
common/phones_mfa.txt. Writes data/russian_mfa_norm.dict.
"""
import unicodedata

IN_PATH = 'data/russian_mfa.dict'
OUT_PATH = 'data/russian_mfa_norm.dict'
PHONES_MFA = '../../common/phones_mfa.txt'
LENGTH_MARK = 'ː'

# The two phones in the MFA dictionary that survive `ː`-stripping but are not in the
# target inventory. They are rare (46 occurrences total); map to the closest target pair.
EXTRA_MAP = {
    'tsʲ': 't̪s̪',
    'dzʲ': 'd̪z̪',
}


def load_target(path):
    with open(path, encoding='utf-8') as fh:
        return {unicodedata.normalize('NFC', line.strip()) for line in fh if line.strip()}


def main():
    target = load_target(PHONES_MFA)
    print(f'target inventory: {len(target)} phones')

    n_lines = 0
    n_changed = 0
    n_mapped = 0
    unknown = {}

    with open(IN_PATH, encoding='utf-8') as fin, \
         open(OUT_PATH, 'w', encoding='utf-8') as fout:
        for line in fin:
            n_lines += 1
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 2:
                fout.write(line)
                continue
            word, phones = parts
            phones_list = phones.split()

            new_list = []
            for p in phones_list:
                orig = p
                p = p.replace(LENGTH_MARK, '')   # Xː → X
                if p in EXTRA_MAP:
                    p = EXTRA_MAP[p]              # tsʲ → t̪s̪, dzʲ → d̪z̪
                    n_mapped += 1
                if p != orig:
                    n_changed += 1
                new_list.append(p)

            for p in new_list:
                p_nfc = unicodedata.normalize('NFC', p)
                if p_nfc not in target and p_nfc != 'spn':
                    unknown[p_nfc] = unknown.get(p_nfc, 0) + 1

            fout.write(f'{word}\t{" ".join(new_list)}\n')

    print(f'lines read: {n_lines}')
    print(f'phones changed (ː removed or mapped): {n_changed}')
    print(f'phones mapped via EXTRA_MAP: {n_mapped}')
    print(f'phones not in target (excluding spn): {len(unknown)}')
    if unknown:
        print('  top-20 unknown phones:')
        for p, n in sorted(unknown.items(), key=lambda x: -x[1])[:20]:
            print(f'    {p!r}: {n}')
    print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
    main()