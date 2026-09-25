"""
bias_gen.py -- file dei BIAS di prova per il testbench di s4d_encoder_ecg_axis

Genera enc_b.mem: il bias del canale ch aggiunge all'uscita il numero del "giro"
del canale, cioe' (ch div 12): 0 per i canali 0..11, 1 per 12..23, ..., A per 120..127.

Il modulo calcola, con F_WENC = 31 (valore di default, SH_OUT = 20):

    y[ch] = ( somma su c di W[ch][c] * x[c]  +  b[ch] ) / 2^20

Il bias viene sommato PRIMA della divisione per 2^20, quindi per far comparire
il valore v in uscita bisogna scrivere nel file  b = v * 2^20.

Insieme a enc_w.mem (vedi weight_gen.py) e allo stimolo 0xTC00, l'uscita vale

    y = 0x0TCR      T = passo + 1
                    C = derivazione letta = ch mod 12
                    R = giro del canale   = ch div 12        ->  ch = 12 * R + C

I pesi mettono le cifre T e C, il bias mette la cifra R: se il bias arrivasse
in ritardo o sul canale sbagliato, la cifra R non tornerebbe con la posizione.

Formato: 128 righe, una per canale, 32 bit con segno = 8 cifre esadecimali.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Parametri del modulo (stessi nomi di s4d_encoder_ecg_axis.v)
# ----------------------------------------------------------------------------
H = 128          # canali di uscita = righe del file
NIN = 12         # derivazioni in ingresso (serve per calcolare il giro del canale)
W_BENC = 32      # bit di un bias (con segno)
F_WENC = 31      # bit frazionari dell'accumulatore (default del modulo)
F_ACT = 11       # bit frazionari dell'uscita

SH_OUT = F_WENC - F_ACT            # 20: di quanti bit viene spostato l'accumulatore
BIAS_UNIT = 1 << SH_OUT            # 2^20: il bias che vale "1" in uscita
BIAS_HEX_DIGITS = W_BENC // 4      # 8 cifre esadecimali per riga
BIAS_MASK = (1 << W_BENC) - 1      # tiene i 32 bit del bias

# il file viene scritto accanto a questo script
OUTPUT_FILE = Path(__file__).with_name("enc_b.mem")


def main():
    file_lines = [f"// bias: aggiunge all'uscita il giro del canale (ch div 12)"
                  f"  (generato da {Path(__file__).name})"]

    for channel in range(H):
        channel_round = channel // NIN       # 0 per ch 0..11, 1 per ch 12..23, ... (cifra R)
        bias = channel_round * BIAS_UNIT     # porta il valore nella scala di F_WENC

        file_lines.append(f"{bias & BIAS_MASK:0{BIAS_HEX_DIGITS}X}  "
                          f"// ch={channel}: aggiunge {channel_round:X} all'uscita")

    # fine riga LF, uguale su Windows e su Linux
    with open(OUTPUT_FILE, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")
    print(f"{OUTPUT_FILE.name}: {H} righe da {BIAS_HEX_DIGITS} cifre esadecimali")


if __name__ == "__main__":
    main()
