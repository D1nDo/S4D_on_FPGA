"""
weight_gen.py -- file dei PESI per il testbench di s4d_core_axis

Per l'encoder e per il decoder i pesi di prova sono numeri semplici, scelti per
dare un'uscita leggibile a occhio. Per il core questo non funziona: ogni layer
contiene filtri con memoria (biquad), due funzioni non lineari a tabella (GELU e
sigmoide) e una matrice 256 x 128, e nessun peso semplice porta a un'uscita che
si riconosce. Qui si usano quindi i PESI VERI della rete addestrata, gli stessi
che vanno sulla scheda: l'uscita viene confrontata con core_model.py.
I pesi veri hanno un vantaggio: sono tutti diversi tra loro, quindi un
coefficiente caricato nel posto sbagliato cambia sempre il risultato.

In questo modulo pesi e bias non sono in file separati (per questo non c'e' un
bias_gen.py): viaggiano tutti insieme in un unico flusso a 64 bit, che il core
riceve su s_axis_w un layer alla volta.

File generati:

w_stream.mem -- il flusso dei pesi dei 6 layer, una parola da 64 bit per riga
    Convertito da ../../board_test/w_stream_ECG.bin (il file che il programma
    di prova manda alla scheda), senza cambiare nessun valore.
    Ogni layer occupa 14977 parole, in quest'ordine (lo stesso di s4d_wload_axis.v):
        1      shift della normalizzazione (6 bit)
        8192   coefficienti dei biquad, 2 parole per biquad (4096 biquad da 80 bit):
               bit [23:0] a1, [47:24] a2, [63:48] b0, [79:64] b1
        128    D e shift del canale: bit [15:0] D, [20:16] shift
        128    guadagno della normalizzazione (16 bit)
        128    bias della normalizzazione (32 bit)
        256    bias del mixing (32 bit)
        6144   pesi del mixing: 512 righe da 12 parole, 64 pesi a 12 bit per riga

gelu_lut.mem, sigmoid_lut.mem -- tabelle delle due funzioni non lineari
    Copiate cosi' come sono da ../../mem: sono le ROM che vanno nel bitstream.
"""

import shutil
import struct
from pathlib import Path

# ----------------------------------------------------------------------------
# Dimensioni del flusso (stessi nomi di s4d_wload_axis.v)
# ----------------------------------------------------------------------------
H = 128                              # canali
NLAYER = 6                           # layer
NUNITS = 8                           # unita' di calcolo dei biquad
LOG2_NB = 9                          # ogni unita' contiene 2^9 = 512 biquad
NMAC = 64                            # pesi del mixing in una riga
W_W = 12                             # bit di un peso del mixing
AXI_DW = 64                          # bit di una parola del flusso

NCOEF = NUNITS * (1 << LOG2_NB)      # 4096 biquad per layer
TH = 2 * H                           # 256 uscite del mixing
NUM_MIX_ROWS = 4 * H                 # 512 righe di pesi del mixing (4 fasi da 128)
BEATS_PER_MIX_ROW = (NMAC * W_W) // AXI_DW   # 12 parole per riga

# quante parole occupa ogni sezione di un layer, nell'ordine in cui arrivano
LAYER_SECTIONS = [
    ("shift della normalizzazione", 1),
    ("coefficienti dei biquad (2 parole per biquad)", 2 * NCOEF),
    ("D e shift per canale", H),
    ("guadagno della normalizzazione", H),
    ("bias della normalizzazione", H),
    ("bias del mixing", TH),
    ("pesi del mixing (12 parole per riga)", NUM_MIX_ROWS * BEATS_PER_MIX_ROW),
]
BEATS_PER_LAYER = sum(section_length for _, section_length in LAYER_SECTIONS)   # 14977

SCRIPT_DIR = Path(__file__).parent
REPOSITORY_DIR = SCRIPT_DIR.parent.parent                       # cartella radice del repository
STREAM_BIN_FILE = REPOSITORY_DIR / "board_test" / "w_stream_ECG.bin"
ROM_DIR = REPOSITORY_DIR / "mem"
LUT_FILES = ["gelu_lut.mem", "sigmoid_lut.mem"]


def main():
    # --- lettura del file binario della scheda: parole da 64 bit, byte meno significativo per primo ---
    raw_bytes = STREAM_BIN_FILE.read_bytes()
    num_beats = len(raw_bytes) // 8
    assert len(raw_bytes) % 8 == 0 and num_beats == NLAYER * BEATS_PER_LAYER, (
        f"{STREAM_BIN_FILE.name}: {len(raw_bytes)} byte, attesi {NLAYER * BEATS_PER_LAYER * 8}")
    beats = struct.unpack(f"<{num_beats}Q", raw_bytes)

    # --- scrittura in esadecimale, con un commento all'inizio di ogni sezione ---
    file_lines = [f"// flusso dei pesi di s4d_core_axis: {NLAYER} layer x {BEATS_PER_LAYER} parole da 64 bit"
                  f"  (generato da {Path(__file__).name} a partire da board_test/{STREAM_BIN_FILE.name})"]
    beat_index = 0
    for layer in range(NLAYER):
        for section_name, section_length in LAYER_SECTIONS:
            file_lines.append(f"// layer {layer}: {section_name}, {section_length} parole")
            for _ in range(section_length):
                file_lines.append(f"{beats[beat_index]:016X}")
                beat_index += 1

    # fine riga LF, uguale su Windows e su Linux
    with open(SCRIPT_DIR / "w_stream.mem", "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")
    print(f"w_stream.mem: {num_beats} parole ({NLAYER} layer x {BEATS_PER_LAYER})")

    # --- le due tabelle sono ROM del bitstream: si copiano identiche ---
    for lut_name in LUT_FILES:
        shutil.copyfile(ROM_DIR / lut_name, SCRIPT_DIR / lut_name)
        print(f"{lut_name}: copiato da mem/")


if __name__ == "__main__":
    main()
