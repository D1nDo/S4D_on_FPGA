"""
weight_gen.py -- file dei PESI di prova per il testbench di s4d_decoder_axis

Il decoder ha due blocchi con pesi, quindi i file sono due:

l6_bn_w.mem -- guadagno della normalizzazione finale (s4d_affine), uno per canale
    Il blocco calcola  a = ( w[ch] * x + b[ch] ) / 2^BN_SH   con BN_SH = 13.
    Quindi il peso 0x2000 = 2^13 vale "x 1" e il peso 0x4000 = 2^14 vale "x 2".
    Qui i canali PARI hanno guadagno 2 e i canali DISPARI guadagno 1: se il
    blocco usasse il guadagno del canale sbagliato, il risultato cambierebbe.
    Formato: 128 righe, 16 bit con segno = 4 cifre esadecimali.

cls_w.mem -- pesi del classificatore (s4d_classifier), 5 per canale
    Il blocco calcola  logit[k] = b[k] + somma su ch di W[ch][k] * xq[ch].
    I primi 125 canali "votano" ciascuno per UNA sola classe, la (ch mod 5),
    con peso 1: ogni classe riceve cosi' il voto di 25 canali esatti.
    I canali 125, 126, 127 avanzano (128 non e' multiplo di 5): hanno peso 1
    su tutte le classi e lo stimolo li tiene a zero, quindi non spostano il
    risultato, ma se ci finisse dentro un valore sbagliato si vedrebbe.
    Formato: 128 righe da 5 pesi a 8 bit = 40 bit = 10 cifre esadecimali,
    peso della classe k nei bit [k*8 +: 8]  (classe 0 nei bit meno significativi).

Le funzioni affine_gain() e classifier_weights() sono usate anche da
gen_decoder_stimulus.py, cosi' la regola e' scritta in un posto solo.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Parametri del modulo (stessi nomi di s4d_decoder_axis.v, valori della scheda ECG)
# ----------------------------------------------------------------------------
H = 128          # canali
K = 5            # classi
BN_W_W = 16      # bit di un guadagno della normalizzazione
BN_SH = 13       # shift della normalizzazione: il peso 2^13 vale "x 1"
W_CLS = 8        # bit di un peso del classificatore

AFFINE_UNITY_WEIGHT = 1 << BN_SH           # 0x2000: guadagno 1
CHANNELS_PER_CLASS = H // K                # 25 canali votano per ogni classe
NUM_VOTING_CHANNELS = CHANNELS_PER_CLASS * K   # 125: i canali 125..127 avanzano

SCRIPT_DIR = Path(__file__).parent         # i file vengono scritti accanto a questo script


def affine_gain(channel):
    """Guadagno della normalizzazione per il canale: 2 sui pari, 1 sui dispari."""
    return 2 if channel % 2 == 0 else 1


def classifier_weights(channel):
    """I 5 pesi del classificatore per il canale (uno per classe).
    Canali 0..124: peso 1 solo sulla classe (channel mod 5).
    Canali 125..127: peso 1 su tutte le classi."""
    if channel < NUM_VOTING_CHANNELS:
        voted_class = channel % K
        return [1 if class_index == voted_class else 0 for class_index in range(K)]
    return [1] * K


def write_mem_file(file_name, file_lines):
    """Scrive un file .mem con fine riga LF (uguale su Windows e su Linux)."""
    with open(SCRIPT_DIR / file_name, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")
    print(f"{file_name}: {len(file_lines) - 1} righe di dati")


def main():
    script_name = Path(__file__).name

    # --- l6_bn_w.mem: guadagno della normalizzazione, un valore a 16 bit per canale ---
    affine_lines = [f"// guadagno della normalizzazione: x2 sui canali pari, x1 sui dispari"
                    f"  (generato da {script_name})"]
    for channel in range(H):
        weight = affine_gain(channel) * AFFINE_UNITY_WEIGHT     # 0x4000 oppure 0x2000
        affine_lines.append(f"{weight:0{BN_W_W // 4}X}  // ch={channel}: guadagno {affine_gain(channel)}")
    write_mem_file("l6_bn_w.mem", affine_lines)

    # --- cls_w.mem: 5 pesi a 8 bit per canale, classe 0 nei bit bassi ---
    weight_mask = (1 << W_CLS) - 1
    classifier_lines = [f"// classificatore: il canale ch vota per la classe (ch mod 5); "
                        f"ch 125..127 su tutte  (generato da {script_name})"]
    for channel in range(H):
        class_weights = classifier_weights(channel)
        row_bits = 0
        for class_index, weight in enumerate(class_weights):
            row_bits |= (weight & weight_mask) << (class_index * W_CLS)   # classe k nei bit [k*8 +: 8]
        classifier_lines.append(f"{row_bits:0{K * W_CLS // 4}X}  // ch={channel}: pesi classi 0..4 = {class_weights}")
    write_mem_file("cls_w.mem", classifier_lines)


if __name__ == "__main__":
    main()
