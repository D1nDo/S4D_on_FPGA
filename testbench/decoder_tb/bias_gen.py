"""
bias_gen.py -- file dei BIAS di prova per il testbench di s4d_decoder_axis

Il decoder ha due blocchi con bias, quindi i file sono due:

l6_bn_b.mem -- bias della normalizzazione finale (s4d_affine), uno per canale
    Il blocco calcola  a = ( w[ch] * x + b[ch] ) / 2^BN_SH   con BN_SH = 13.
    Il bias e' sommato PRIMA della divisione: per aggiungere v all'uscita
    bisogna scrivere  b = v * 2^13.
    Qui il bias del canale ch aggiunge 16 * ch. Lo stimolo contiene lo stesso
    valore col segno opposto, quindi il blocco lo TOGLIE: e' quello che fa una
    normalizzazione vera (toglie a ogni canale il suo valore medio).
    Se il bias arrivasse sul canale sbagliato, la cancellazione non tornerebbe.
    In piu' ogni bias contiene MEZZA unita' (2^12): il risultato esatto finisce
    sempre in ",5" e l'arrotondamento del blocco lo porta all'intero sopra.
    Serve a provare l'arrotondamento: se mancasse, ogni campione uscirebbe
    piu' basso di 1 (lo rivela la sequenza C di gen_decoder_stimulus.py).
    Formato: 128 righe, 32 bit con segno = 8 cifre esadecimali.

cls_b.mem -- bias del classificatore (s4d_classifier), uno per classe
    Il blocco calcola  logit[k] = b[k] + somma su ch di W[ch][k] * xq[ch].
    Qui b[k] = k: l'ultima cifra del logit dice di quale classe e'.
    Formato: 5 righe, 32 bit con segno = 8 cifre esadecimali.

Le funzioni affine_offset() e classifier_bias() sono usate anche da
gen_decoder_stimulus.py, cosi' la regola e' scritta in un posto solo.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Parametri del modulo (stessi nomi di s4d_decoder_axis.v, valori della scheda ECG)
# ----------------------------------------------------------------------------
H = 128          # canali
K = 5            # classi
BN_W_B = 32      # bit di un bias della normalizzazione
BN_SH = 13       # shift della normalizzazione: il bias 2^13 aggiunge 1 all'uscita
W_ACC = 32       # bit di un bias del classificatore

AFFINE_BIAS_UNIT = 1 << BN_SH              # 2^13: il bias che aggiunge 1 all'uscita
AFFINE_HALF_UNIT = 1 << (BN_SH - 1)        # 2^12: mezza unita', messa in ogni bias
AFFINE_ROUNDING_STEP = 1                   # di quanto l'arrotondamento alza l'uscita per via della mezza unita'
BIAS_MASK = (1 << 32) - 1                  # per scrivere i 32 bit in esadecimale
OFFSET_PER_CHANNEL = 16                    # il bias del canale ch aggiunge 16 * ch

SCRIPT_DIR = Path(__file__).parent         # i file vengono scritti accanto a questo script


def affine_offset(channel):
    """Valore intero che il bias della normalizzazione aggiunge all'uscita del canale
    (la mezza unita' in piu' diventa AFFINE_ROUNDING_STEP dopo l'arrotondamento)."""
    return OFFSET_PER_CHANNEL * channel


def classifier_bias(class_index):
    """Bias del classificatore per la classe: il numero stesso della classe."""
    return class_index


def write_mem_file(file_name, file_lines):
    """Scrive un file .mem con fine riga LF (uguale su Windows e su Linux)."""
    with open(SCRIPT_DIR / file_name, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")
    print(f"{file_name}: {len(file_lines) - 1} righe di dati")


def main():
    script_name = Path(__file__).name

    # --- l6_bn_b.mem: bias della normalizzazione, un valore a 32 bit per canale ---
    affine_lines = [f"// bias della normalizzazione: il canale ch aggiunge 16*ch + mezza unita' all'uscita"
                    f"  (generato da {script_name})"]
    for channel in range(H):
        # parte intera nella scala di BN_SH, piu' mezza unita' per far lavorare l'arrotondamento
        bias = affine_offset(channel) * AFFINE_BIAS_UNIT + AFFINE_HALF_UNIT
        affine_lines.append(f"{bias & BIAS_MASK:0{BN_W_B // 4}X}  "
                            f"// ch={channel}: aggiunge {affine_offset(channel)},5")
    write_mem_file("l6_bn_b.mem", affine_lines)

    # --- cls_b.mem: bias del classificatore, un valore a 32 bit per classe ---
    classifier_lines = [f"// bias del classificatore: b[k] = k  (generato da {script_name})"]
    for class_index in range(K):
        bias = classifier_bias(class_index)
        classifier_lines.append(f"{bias & BIAS_MASK:0{W_ACC // 4}X}  // classe {class_index}: aggiunge {bias}")
    write_mem_file("cls_b.mem", classifier_lines)


if __name__ == "__main__":
    main()
