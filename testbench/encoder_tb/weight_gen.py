"""
weight_gen.py -- file dei PESI di prova per il testbench di s4d_encoder_ecg_axis

Genera enc_w.mem: pesi "a selettore". Il canale di uscita ch guarda UNA SOLA
derivazione, la numero (ch mod 12), con peso 0x10000; gli altri 11 pesi sono 0.

Il modulo calcola, con F_WENC = 31 (valore di default, SH_OUT = 20):

    y[ch] = ( somma su c di W[ch][c] * x[c]  +  b[ch] ) / 2^20

Il peso 0x10000 = 2^16 da' un guadagno esatto di 2^16 / 2^20 = 1/16, quindi il
campione esce spostato di UNA CIFRA ESADECIMALE verso destra: 0x1B00 -> 0x01B0.

Insieme a enc_b.mem (vedi bias_gen.py) e allo stimolo 0xTC00, l'uscita vale

    y = 0x0TCR      T = passo + 1
                    C = derivazione letta = ch mod 12
                    R = giro del canale   = ch div 12        ->  ch = 12 * R + C

Formato di una riga (quello che legge $readmemh nel modulo):
    12 pesi da 18 bit = 216 bit = 54 cifre esadecimali,
    peso della derivazione c nei bit [c*18 +: 18]  (c0 nei bit meno significativi).
18 bit non sono un multiplo di 4, quindi i pesi NON sono allineati alle cifre
esadecimali: per questo accanto a ogni riga c'e' un commento con il peso in chiaro.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Parametri del modulo (stessi nomi di s4d_encoder_ecg_axis.v)
# ----------------------------------------------------------------------------
H = 128          # canali di uscita = righe del file
NIN = 12         # derivazioni in ingresso = pesi per riga
W_WENC = 18      # bit di un peso (con segno)

ROW_BITS = NIN * W_WENC          # 216 bit per riga
ROW_HEX_DIGITS = ROW_BITS // 4   # 54 cifre esadecimali per riga
WEIGHT_MASK = (1 << W_WENC) - 1  # tiene i 18 bit del peso

# peso della derivazione selezionata: 0x10000, guadagno 1/16 con SH_OUT = 20
SELECTED_LEAD_WEIGHT = 1 << 16

# il file viene scritto accanto a questo script
OUTPUT_FILE = Path(__file__).with_name("enc_w.mem")


def pack_weight_row(lead_weights):
    """Unisce i 12 pesi di un canale in un'unica parola da 216 bit.
    Il peso della derivazione c viene messo nei bit [c*18 +: 18]."""
    row_bits = 0
    for lead, weight in enumerate(lead_weights):
        row_bits |= (weight & WEIGHT_MASK) << (lead * W_WENC)
    return row_bits


def main():
    file_lines = [f"// pesi a selettore: canale ch = derivazione (ch mod 12) / 16"
                  f"  (generato da {Path(__file__).name})"]

    for channel in range(H):
        selected_lead = channel % NIN        # l'unica derivazione che questo canale guarda

        # 12 pesi del canale: 0x10000 sulla derivazione scelta, 0 sulle altre
        lead_weights = [SELECTED_LEAD_WEIGHT if lead == selected_lead else 0
                        for lead in range(NIN)]

        row_bits = pack_weight_row(lead_weights)
        file_lines.append(f"{row_bits:0{ROW_HEX_DIGITS}X}  "
                          f"// ch={channel}: c{selected_lead}=0x{SELECTED_LEAD_WEIGHT:05X}")

    # fine riga LF, uguale su Windows e su Linux
    with open(OUTPUT_FILE, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")
    print(f"{OUTPUT_FILE.name}: {H} righe da {ROW_HEX_DIGITS} cifre esadecimali")


if __name__ == "__main__":
    main()
