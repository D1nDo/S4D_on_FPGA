"""
core_model.py -- modello di riferimento di s4d_core_axis

Rifa' in Python, con numeri interi, gli stessi calcoli del core e scrive i
campioni che il core DEVE produrre. Il testbench confronta l'uscita dell'RTL
con questo file, campione per campione.

Il core applica 6 layer uno dopo l'altro alla stessa sequenza. Un layer, per
ogni passo temporale (128 canali), fa:

  1. s4d_affine        u[ch]   = normalizzazione: guadagno e bias per canale
  2. s4d_biquad_bank   ssm[ch] = D[ch]*u[ch] + somma di 32 filtri biquad sul canale
                                 (i filtri hanno memoria: ricordano i passi precedenti)
  3. s4d_pwl (GELU)    g[ch]   = GELU(ssm[ch]), letta da una tabella con interpolazione
  4. s4d_mixing_glu    a[c], gate[c] = matrice 256 x 128 applicata a g, piu' bias
                       y[ch]   = a[ch] * sigmoide(gate[ch])
  5. s4d_layer         out[ch] = x[ch] + y[ch]      (somma con l'ingresso del layer)

All'inizio di ogni layer la memoria dei filtri riparte da zero.

Il modello legge GLI STESSI file che legge l'RTL (stimolo, flusso dei pesi,
tabelle GELU e sigmoide), quindi verifica anche che i formati siano
interpretati allo stesso modo. In particolare decodifica il flusso dei pesi
con le stesse regole di s4d_wload_axis.v.

File scritto: un campione a 16 bit per riga (4 cifre esadecimali), nell'ordine
in cui il core li emette. Il modello stampa anche il valore atteso di
status_err (bit 3 = saturazione nei filtri), da passare al testbench.

Uso (i valori tra parentesi quadre sono quelli di default):
    python3 core_model.py [--stim stim_2x8x128.mem] [--weights w_stream.mem]
                          [--out expected_2x8x128.mem]
"""

import argparse
from pathlib import Path

# ----------------------------------------------------------------------------
# Parametri del modulo (stessi nomi di s4d_core_axis.v)
#   F_SSM = 9   : valore del progetto ECG (impostato nel block design)
#   L, LOG2_L   : RIDOTTI per la prova (in scheda L = 1024, LOG2_L = 10)
# ----------------------------------------------------------------------------
H = 128              # canali
L = 8                # passi temporali per sequenza (prova)
NLAYER = 6           # layer
NUNITS = 8           # unita' di calcolo dei biquad
NCH = 16             # canali per unita'
MODES = 32           # biquad per canale
LOG2_NB = 9          # log2(NCH * MODES): biquad per unita'
NMAC = 64            # moltiplicatori del mixing
TH = 2 * H           # uscite del mixing: 128 valori "a" + 128 valori "gate"
AXI_DW = 64          # bit di una parola del flusso dei pesi

W_ACT, F_ACT = 16, 11        # ingresso e uscita del layer
W_U, F_U = 16, 11            # uscita della normalizzazione
W_A, F_A = 24, 22            # coefficienti a1, a2 dei biquad
W_B, F_B = 16, 15            # coefficienti b0, b1 dei biquad
W_S, F_Y = 32, 19            # stato e uscita di un biquad
W_D, F_D = 16, 13            # coefficiente D
W_BSH = 5                    # bit dello shift per canale
W_BACC = 40                  # accumulatore dei 32 biquad di un canale
W_SSM, F_SSM = 16, 9         # uscita del banco di biquad
W_W, F_W = 12, 11            # pesi del mixing
W_MACC = 32                  # accumulatore del mixing
F_SIGI, F_SIGO = 11, 14      # ingresso e uscita della sigmoide
BN_W_W, BN_W_B, BN_W_ACC = 16, 32, 34   # normalizzazione: guadagno, bias, accumulatore

# shift ricavati dai formati, come i localparam dell'RTL
PW = W_A + W_S                       # 56: larghezza dei conti interni di un biquad
SH_A = F_A                           # 22: dopo il prodotto a * y
SH_D = F_D + F_U - F_Y               # 5:  dopo il prodotto D * u
SH_SSM = F_Y - F_SSM                 # 10: uscita del banco di biquad
FACC = F_W + F_SSM                   # 20: bit frazionari dell'accumulatore del mixing
RQ_A = FACC - F_ACT                  # 9:  riduzione di "a" a 16 bit
RQ_G = FACC - F_SIGI                 # 9:  riduzione di "gate" a 16 bit
SH_GLU = F_ACT + F_SIGO - F_ACT      # 14: dopo il prodotto a * sigmoide

# tabelle a tratti (s4d_pwl): 256 segmenti sull'intervallo da -8 a +8
PWL_XH = 3
PWL_LOG2_NSEG = 8

# sezioni del flusso dei pesi di un layer (s4d_wload_axis.v)
NCOEF = NUNITS * (1 << LOG2_NB)                  # 4096 biquad
NUM_MIX_ROWS = (TH // NMAC) * H                  # 512 righe di pesi del mixing
BEATS_PER_MIX_ROW = (NMAC * W_W) // AXI_DW       # 12 parole per riga
COEF_BEGIN = 1
COEF_END = COEF_BEGIN + 2 * NCOEF
D_END = COEF_END + H
BN_W_END = D_END + H
BN_B_END = BN_W_END + H
MIX_BIAS_END = BN_B_END + TH
BEATS_PER_LAYER = MIX_BIAS_END + NUM_MIX_ROWS * BEATS_PER_MIX_ROW    # 14977

SCRIPT_DIR = Path(__file__).parent   # cartella in cui si cercano i file .mem


# ----------------------------------------------------------------------------
# Funzioni di aritmetica a larghezza fissa
# ----------------------------------------------------------------------------
def to_signed(raw_value, num_bits):
    """Interpreta i num_bits meno significativi di raw_value come numero con segno
    in complemento a due (cosi' si riproduce anche il "giro" di un registro pieno)."""
    raw_value &= (1 << num_bits) - 1
    sign_bit = 1 << (num_bits - 1)
    return raw_value - (1 << num_bits) if raw_value & sign_bit else raw_value


def round_shift_right(value, shift, register_bits):
    """(value + 2^(shift-1)) >> shift, con la somma fatta in un registro di
    register_bits bit, come '(v + RND) >>> s' nell'RTL. Con shift = 0 non cambia nulla."""
    if shift == 0:
        return value
    return to_signed(value + (1 << (shift - 1)), register_bits) >> shift


def saturate(value, num_bits):
    """Limita value a un numero con segno di num_bits bit.
    Restituisce (valore, True se e' stato tagliato)."""
    highest = (1 << (num_bits - 1)) - 1
    lowest = -(1 << (num_bits - 1))
    if value > highest:
        return highest, True
    if value < lowest:
        return lowest, True
    return value, False


# ----------------------------------------------------------------------------
# Lettura dei file .mem
# ----------------------------------------------------------------------------
def read_mem_file(mem_path, expected_rows=None):
    """Legge un file per $readmemh: toglie i commenti // e le righe vuote."""
    values = []
    for text_line in mem_path.read_text().splitlines():
        hex_text = text_line.split("//")[0].strip()
        if hex_text:
            values.append(int(hex_text, 16))
    if expected_rows is not None and len(values) != expected_rows:
        raise ValueError(f"{mem_path.name}: {len(values)} righe, attese {expected_rows}")
    return values


def load_stimulus(stimulus_path):
    """Restituisce sequences[s][t][ch]: campioni con segno a 16 bit."""
    samples = [to_signed(raw, W_ACT) for raw in read_mem_file(stimulus_path)]
    samples_per_sequence = L * H
    if len(samples) % samples_per_sequence != 0:
        raise ValueError(f"{stimulus_path.name}: {len(samples)} campioni, non multiplo di {samples_per_sequence}")
    return [[samples[start + timestep * H: start + (timestep + 1) * H] for timestep in range(L)]
            for start in range(0, len(samples), samples_per_sequence)]


def load_lookup_table(table_path):
    """Tabella a tratti: ogni riga a 32 bit contiene la pendenza (16 bit alti)
    e il valore di partenza del segmento (16 bit bassi), entrambi con segno."""
    rows = read_mem_file(table_path, 1 << PWL_LOG2_NSEG)
    return {"base": [to_signed(row, 16) for row in rows],
            "slope": [to_signed(row >> 16, 16) for row in rows]}


def decode_layer_weights(beats):
    """Ricostruisce i pesi di un layer dalle sue 14977 parole da 64 bit,
    con le stesse regole di s4d_wload_axis.v."""
    layer = {}

    # parola 0: shift della normalizzazione
    layer["bn_shift"] = beats[0] & 0x3F

    # 4096 biquad da 80 bit, 2 parole ciascuno: prima i 64 bit bassi, poi i 16 alti.
    # Ordine: unita' (8), poi canale dentro l'unita' (16), poi modo (32)
    # -> il biquad numero e appartiene al canale e // 32, modo e % 32.
    layer["a1"] = [[0] * MODES for _ in range(H)]
    layer["a2"] = [[0] * MODES for _ in range(H)]
    layer["b0"] = [[0] * MODES for _ in range(H)]
    layer["b1"] = [[0] * MODES for _ in range(H)]
    for entry in range(NCOEF):
        low_64_bits = beats[COEF_BEGIN + 2 * entry]
        high_16_bits = beats[COEF_BEGIN + 2 * entry + 1] & 0xFFFF
        coefficient_word = (high_16_bits << 64) | low_64_bits
        channel, mode = entry // MODES, entry % MODES
        layer["a1"][channel][mode] = to_signed(coefficient_word, W_A)                    # bit [23:0]
        layer["a2"][channel][mode] = to_signed(coefficient_word >> W_A, W_A)             # bit [47:24]
        layer["b0"][channel][mode] = to_signed(coefficient_word >> (2 * W_A), W_B)       # bit [63:48]
        layer["b1"][channel][mode] = to_signed(coefficient_word >> (2 * W_A + W_B), W_B) # bit [79:64]

    # 128 parole: D (16 bit con segno) e shift del canale (5 bit)
    layer["d"] = [to_signed(beat, W_D) for beat in beats[COEF_END:D_END]]
    layer["channel_shift"] = [(beat >> W_D) & ((1 << W_BSH) - 1) for beat in beats[COEF_END:D_END]]

    # normalizzazione: 128 guadagni a 16 bit, poi 128 bias a 32 bit
    layer["bn_weight"] = [to_signed(beat, BN_W_W) for beat in beats[D_END:BN_W_END]]
    layer["bn_bias"] = [to_signed(beat, BN_W_B) for beat in beats[BN_W_END:BN_B_END]]

    # mixing: 256 bias a 32 bit
    layer["mix_bias"] = [to_signed(beat, W_MACC) for beat in beats[BN_B_END:MIX_BIAS_END]]

    # mixing: 512 righe da 768 bit (12 parole, la prima contiene i bit bassi).
    # La riga (fase * 128 + h) contiene 64 pesi a 12 bit: il peso nella posizione m
    # serve all'uscita c = fase * 64 + m e moltiplica l'ingresso h.
    layer["mix_weight"] = [[0] * H for _ in range(TH)]
    for row in range(NUM_MIX_ROWS):
        row_bits = 0
        for beat_in_row in range(BEATS_PER_MIX_ROW):
            row_bits |= beats[MIX_BIAS_END + row * BEATS_PER_MIX_ROW + beat_in_row] << (AXI_DW * beat_in_row)
        phase, input_channel = row // H, row % H
        for mac in range(NMAC):
            layer["mix_weight"][phase * NMAC + mac][input_channel] = to_signed(row_bits >> (mac * W_W), W_W)

    return layer


def load_weight_stream(stream_path):
    """Restituisce la lista dei 6 layer decodificati."""
    beats = read_mem_file(stream_path, NLAYER * BEATS_PER_LAYER)
    return [decode_layer_weights(beats[layer * BEATS_PER_LAYER:(layer + 1) * BEATS_PER_LAYER])
            for layer in range(NLAYER)]


# ----------------------------------------------------------------------------
# I blocchi di un layer
# ----------------------------------------------------------------------------
def affine(sample, weight, bias, shift):
    """s4d_affine: (peso * x + bias) / 2^shift con arrotondamento, saturazione a 16 bit."""
    accumulator = to_signed(weight * sample + bias, BN_W_ACC)
    return saturate(round_shift_right(accumulator, shift, BN_W_ACC), W_U)[0]


def shift_and_saturate(value, shift):
    """Funzione 'satw' di s4d_biquad_unit.v: arrotonda e sposta su 56 bit, poi satura a 32 bit."""
    return saturate(round_shift_right(value, shift, PW), W_S)


class BiquadBank:
    """s4d_biquad_bank: 32 biquad per canale. Tiene la memoria (s1, s2) di ogni biquad."""

    def __init__(self):
        self.state_1 = [[0] * MODES for _ in range(H)]
        self.state_2 = [[0] * MODES for _ in range(H)]
        self.overflow = False            # diventa True se un conto interno satura (flag layer_ovf)

    def clear(self):
        """Inizio di una sequenza: la memoria dei filtri riparte da zero."""
        self.state_1 = [[0] * MODES for _ in range(H)]
        self.state_2 = [[0] * MODES for _ in range(H)]

    def step(self, normalized_samples, layer):
        """Un passo temporale: restituisce i 128 valori ssm."""
        outputs = []
        for channel in range(H):
            u = normalized_samples[channel]
            channel_shift = layer["channel_shift"][channel]

            # ramo diretto D * u, portato al formato dell'uscita dei biquad
            direct_term, saturated = shift_and_saturate(layer["d"][channel] * u, SH_D)
            self.overflow |= saturated

            accumulator = direct_term
            for mode in range(MODES):
                state_1 = self.state_1[channel][mode]
                state_2 = self.state_2[channel][mode]

                # uscita del biquad: y = (b0*u + s1 * 2^shift) / 2^shift
                shifted_state = to_signed(state_1 << channel_shift, PW)
                y, saturated_y = shift_and_saturate(
                    to_signed(layer["b0"][channel][mode] * u + shifted_state, PW), channel_shift)

                # termini per il nuovo stato
                b1_term, saturated_b1 = shift_and_saturate(layer["b1"][channel][mode] * u, channel_shift)
                a1_term, saturated_a1 = shift_and_saturate(layer["a1"][channel][mode] * y, SH_A)
                a2_term, saturated_a2 = shift_and_saturate(layer["a2"][channel][mode] * y, SH_A)

                # nuovo stato: s1 = b1*u - a1*y + s2 ,  s2 = -a2*y
                new_state_1, saturated_s1 = shift_and_saturate(b1_term - a1_term + state_2, 0)
                self.state_1[channel][mode] = new_state_1
                self.state_2[channel][mode] = to_signed(-a2_term, W_S)

                self.overflow |= (saturated_y or saturated_b1 or saturated_a1
                                  or saturated_a2 or saturated_s1)

                # somma delle 32 uscite del canale, in un registro a 40 bit
                accumulator = to_signed(accumulator + y, W_BACC)

            # uscita del canale: arrotondamento, shift e saturazione a 16 bit
            outputs.append(saturate(round_shift_right(accumulator, SH_SSM, W_BACC), W_SSM)[0])
        return outputs


def piecewise_linear(x, table, input_fraction_bits, pass_input_above, value_below, value_above):
    """s4d_pwl: funzione letta da una tabella di 256 segmenti con interpolazione lineare.
    Fuori dall'intervallo [-8, +8) non usa la tabella:
      sotto  -> value_below
      sopra  -> l'ingresso stesso (pass_input_above) oppure value_above"""
    fraction_width = input_fraction_bits + PWL_XH + 1 - PWL_LOG2_NSEG   # bit dentro un segmento
    offset = 1 << (PWL_XH + input_fraction_bits)                         # 8.0 nel formato dell'ingresso

    if x < -offset:
        return value_below
    if x >= offset:
        return x if pass_input_above else value_above

    shifted_x = x + offset                                # da 0 a 2*offset - 1
    segment = shifted_x >> fraction_width                 # quale dei 256 segmenti
    fraction = shifted_x & ((1 << fraction_width) - 1)    # posizione dentro il segmento

    interpolated = table["base"][segment] + (
        (table["slope"][segment] * fraction + (1 << (fraction_width - 1))) >> fraction_width)
    return saturate(interpolated, 16)[0]


def mixing_glu(gelu_outputs, layer, sigmoid_table):
    """s4d_mixing_glu: matrice 256 x 128, poi y[ch] = a[ch] * sigmoide(gate[ch])."""
    # 256 somme pesate: si parte dal bias e si aggiunge un prodotto per ingresso (registro a 32 bit)
    accumulators = []
    for output_index in range(TH):
        accumulator = layer["mix_bias"][output_index]
        weights = layer["mix_weight"][output_index]
        for input_channel in range(H):
            accumulator = to_signed(accumulator + weights[input_channel] * gelu_outputs[input_channel], W_MACC)
        accumulators.append(accumulator)

    outputs = []
    for channel in range(H):
        # le prime 128 somme sono i valori "a", le altre 128 i "gate": riduzione a 16 bit
        a_value = saturate(round_shift_right(accumulators[channel], RQ_A, W_MACC), 16)[0]
        gate_value = saturate(round_shift_right(accumulators[H + channel], RQ_G, W_MACC), 16)[0]

        # sigmoide: 0 sotto -8, esattamente 1.0 (= 2^14) sopra +8
        sigmoid_value = piecewise_linear(gate_value, sigmoid_table, F_SIGI,
                                         pass_input_above=False, value_below=0, value_above=1 << F_SIGO)

        # prodotto a * sigmoide, riportato al formato dell'uscita
        outputs.append(saturate(round_shift_right(a_value * sigmoid_value, SH_GLU, 32), W_ACT)[0])
    return outputs


def run_layer(input_frames, layer, gelu_table, sigmoid_table, biquad_bank):
    """Applica un layer a tutta la sequenza. Restituisce i passi in uscita."""
    biquad_bank.clear()                                   # la memoria dei filtri riparte da zero
    output_frames = []
    for frame in input_frames:
        normalized = [affine(frame[channel], layer["bn_weight"][channel], layer["bn_bias"][channel],
                             layer["bn_shift"]) for channel in range(H)]
        ssm = biquad_bank.step(normalized, layer)
        # GELU: 0 sotto -8, uguale all'ingresso sopra +8
        gelu_outputs = [piecewise_linear(value, gelu_table, F_SSM,
                                         pass_input_above=True, value_below=0, value_above=0)
                        for value in ssm]
        glu_outputs = mixing_glu(gelu_outputs, layer, sigmoid_table)
        # somma con l'ingresso del layer, saturata a 16 bit
        output_frames.append([saturate(frame[channel] + glu_outputs[channel], W_ACT)[0]
                              for channel in range(H)])
    return output_frames


def main():
    parser = argparse.ArgumentParser(description="Modello di riferimento del core S4D")
    parser.add_argument("--stim", default="stim_2x8x128.mem", help="file dello stimolo")
    parser.add_argument("--weights", default="w_stream.mem", help="flusso dei pesi dei 6 layer")
    parser.add_argument("--gelu", default="gelu_lut.mem", help="tabella della GELU")
    parser.add_argument("--sigmoid", default="sigmoid_lut.mem", help="tabella della sigmoide")
    parser.add_argument("--out", default="expected_2x8x128.mem", help="file dei campioni attesi")
    args = parser.parse_args()

    sequences = load_stimulus(SCRIPT_DIR / args.stim)
    layers = load_weight_stream(SCRIPT_DIR / args.weights)
    gelu_table = load_lookup_table(SCRIPT_DIR / args.gelu)
    sigmoid_table = load_lookup_table(SCRIPT_DIR / args.sigmoid)

    sample_mask = (1 << W_ACT) - 1
    file_lines = [
        f"// campioni attesi di s4d_core_axis (generato da {Path(__file__).name})",
        f"// stimolo {args.stim}, pesi {args.weights}, L {L}, F_SSM {F_SSM}, {NLAYER} layer",
    ]

    biquad_bank = BiquadBank()           # uno solo per tutta la prova: il flag di saturazione resta alzato
    for sequence_index, input_frames in enumerate(sequences):
        frames = input_frames
        for layer in layers:
            frames = run_layer(frames, layer, gelu_table, sigmoid_table, biquad_bank)

        saturated_outputs = 0
        for timestep, frame in enumerate(frames):
            for channel, sample in enumerate(frame):
                file_lines.append(f"{sample & sample_mask:04X}  "
                                  f"// sequenza {sequence_index} t={timestep} ch={channel} ({sample})")
                saturated_outputs += sample in (-(1 << (W_ACT - 1)), (1 << (W_ACT - 1)) - 1)

        all_samples = [sample for frame in frames for sample in frame]
        print(f"sequenza {sequence_index}: uscita tra {min(all_samples)} e {max(all_samples)}, "
              f"{saturated_outputs} campioni a fondo scala, "
              f"saturazione nei filtri finora: {'si' if biquad_bank.overflow else 'no'}")

    # status_err = {err_bank, layer_ovf, err_ovw, err_enc_len, err_out_ovr}: atteso solo layer_ovf (bit 3)
    expected_status = (1 << 3) if biquad_bank.overflow else 0
    file_lines.insert(2, f"// status_err atteso a fine prova: {expected_status:05b}")

    # fine riga LF, uguale su Windows e su Linux
    with open(SCRIPT_DIR / args.out, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")

    print(f"{args.out}: {len(sequences) * L * H} campioni ({len(sequences)} sequenze x {L} passi x {H} canali)")
    print(f"status_err atteso a fine prova: {expected_status:05b}  "
          f"(bit 3 = saturazione nei filtri)")


if __name__ == "__main__":
    main()
