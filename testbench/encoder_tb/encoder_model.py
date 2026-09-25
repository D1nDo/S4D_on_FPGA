"""
encoder_model.py -- modello di riferimento di s4d_encoder_ecg_axis

Rifa' in Python, con numeri interi, gli stessi calcoli del modulo Verilog e
scrive i valori che il modulo DEVE produrre. Il testbench confronta l'uscita
dell'RTL con questo file, dato per dato.

Calcolo, per ogni passo temporale t e per ogni canale di uscita ch (0..127):

    somma  = W[ch][0]*x[t][0] + W[ch][1]*x[t][1] + ... + W[ch][11]*x[t][11]
    acc    = somma + b[ch]
    y      = satura16( arrotonda( acc / 2^SH_OUT ) )      con SH_OUT = F_WENC - F_ACT

Il modello legge GLI STESSI file che legge l'RTL (stimolo, pesi, bias), quindi
verifica anche che il formato dei .mem sia interpretato allo stesso modo.

File letti:
    stimolo   una parola AXI da 32 bit per riga
              bit [15:0] = derivazione pari, bit [31:16] = derivazione dispari
    pesi      128 righe da 12 x 18 bit (54 cifre esadecimali),
              peso della derivazione c0 nei 18 bit meno significativi
    bias      128 righe da 32 bit, nella scala di F_WENC

File scritto:
    atteso    un valore da 16 bit per riga (4 cifre esadecimali), nell'ordine in
              cui il modulo li emette: passo 0 canali 0..127, passo 1 canali 0..127, ...

Uso (i valori tra parentesi quadre sono quelli di default):
    python3 encoder_model.py [--stim stim_10x12.mem] [--weights enc_w.mem]
                             [--bias enc_b.mem] [--out expected_10x128.mem]
                             [--f-wenc 31]
"""

import argparse
from pathlib import Path

# ----------------------------------------------------------------------------
# Parametri del modulo (stessi nomi e stessi valori di s4d_encoder_ecg_axis.v)
# ----------------------------------------------------------------------------
H = 128              # canali di uscita
NIN = 12             # derivazioni in ingresso
W_X = 16             # bit di un campione di ingresso (Q5.11 con segno)
W_WENC = 18          # bit di un peso
W_BENC = 32          # bit di un bias
W_ENCACC = 40        # bit dell'accumulatore
W_ACT = 16           # bit dell'uscita
F_ACT = 11           # bit frazionari dell'uscita
F_WENC_DEFAULT = 31  # bit frazionari dell'accumulatore (default del modulo)

SAMPLES_PER_WORD = 2                     # campioni dentro una parola AXI da 32 bit
WORDS_PER_TIMESTEP = NIN // SAMPLES_PER_WORD   # 6 parole per passo temporale

# cartella in cui si trova questo script: i file .mem si cercano qui
SCRIPT_DIR = Path(__file__).parent


# ----------------------------------------------------------------------------
# Funzioni di aritmetica a larghezza fissa
# ----------------------------------------------------------------------------
def to_signed(raw_value, num_bits):
    """Interpreta i num_bits meno significativi di raw_value come numero con segno
    in complemento a due (e' quello che fa Verilog con un 'wire signed')."""
    raw_value &= (1 << num_bits) - 1             # tiene solo num_bits bit
    sign_bit = 1 << (num_bits - 1)               # peso del bit di segno
    return raw_value - (1 << num_bits) if raw_value & sign_bit else raw_value


def round_shift_right(value, shift):
    """Divide per 2^shift arrotondando al piu' vicino (le meta' vanno verso l'alto).

    Stessa formula dell'RTL: (acc + RND) >>> SH_OUT, con RND = 2^(SH_OUT-1).
    In Python '>>' su un intero negativo arrotonda verso il basso, come '>>>'
    con segno in Verilog.
    """
    if shift == 0:
        return value                             # nessuno shift: RND vale 0 nell'RTL
    rounding_constant = 1 << (shift - 1)         # mezza unita' dell'ultimo bit tenuto
    return (value + rounding_constant) >> shift


def saturate(value, num_bits):
    """Limita value all'intervallo di un numero con segno a num_bits bit.
    Restituisce anche True se il valore e' stato tagliato (serve per sat_sticky)."""
    highest = (1 << (num_bits - 1)) - 1          # +32767 per 16 bit
    lowest = -(1 << (num_bits - 1))              # -32768 per 16 bit
    if value > highest:
        return highest, True
    if value < lowest:
        return lowest, True
    return value, False


# ----------------------------------------------------------------------------
# Lettura dei file .mem
# ----------------------------------------------------------------------------
def read_mem_file(mem_path):
    """Legge un file per $readmemh e restituisce la lista dei valori interi.
    Toglie i commenti che iniziano con // e salta le righe vuote."""
    values = []
    for text_line in mem_path.read_text().splitlines():
        hex_text = text_line.split("//")[0].strip()    # parte prima del commento
        if hex_text:
            values.append(int(hex_text, 16))
    return values


def load_stimulus(stimulus_path):
    """Restituisce una lista di passi temporali; ogni passo e' una lista di
    12 campioni con segno, nell'ordine c0..c11."""
    axi_words = read_mem_file(stimulus_path)
    if len(axi_words) % WORDS_PER_TIMESTEP != 0:
        raise ValueError(f"{stimulus_path.name}: {len(axi_words)} parole, "
                         f"non multiplo di {WORDS_PER_TIMESTEP}")

    timesteps = []
    for first_word in range(0, len(axi_words), WORDS_PER_TIMESTEP):
        lead_samples = []
        for axi_word in axi_words[first_word:first_word + WORDS_PER_TIMESTEP]:
            even_lead_sample = to_signed(axi_word, W_X)           # bit [15:0]
            odd_lead_sample = to_signed(axi_word >> W_X, W_X)     # bit [31:16]
            lead_samples += [even_lead_sample, odd_lead_sample]
        timesteps.append(lead_samples)
    return timesteps


def load_weights(weight_path):
    """Restituisce weights[ch][c]: 128 righe da 12 pesi con segno.
    Il peso della derivazione c sta nei bit [c*18 +: 18] della riga (C0_AT_LSB = 1)."""
    weight_rows = read_mem_file(weight_path)
    if len(weight_rows) != H:
        raise ValueError(f"{weight_path.name}: {len(weight_rows)} righe, attese {H}")
    return [[to_signed(row_bits >> (lead * W_WENC), W_WENC) for lead in range(NIN)]
            for row_bits in weight_rows]


def load_biases(bias_path):
    """Restituisce biases[ch]: 128 valori con segno a 32 bit."""
    bias_words = read_mem_file(bias_path)
    if len(bias_words) != H:
        raise ValueError(f"{bias_path.name}: {len(bias_words)} righe, attese {H}")
    return [to_signed(bias_word, W_BENC) for bias_word in bias_words]


# ----------------------------------------------------------------------------
# Il calcolo dell'encoder
# ----------------------------------------------------------------------------
def encode_timestep(lead_samples, weights, biases, shift_out):
    """Calcola i 128 canali di uscita di un passo temporale.
    Restituisce (lista dei 128 valori, numero di canali saturati)."""
    channel_outputs = []
    saturated_channels = 0

    for channel in range(H):
        # stadi S2-S4 dell'RTL: 12 prodotti e la loro somma
        weighted_sum = sum(weights[channel][lead] * lead_samples[lead]
                           for lead in range(NIN))

        # stadio S5: somma del bias; l'accumulatore dell'RTL e' a 40 bit
        accumulator = to_signed(weighted_sum + biases[channel], W_ENCACC)

        # stadio S6: arrotondamento, shift e saturazione a 16 bit
        rounded = round_shift_right(accumulator, shift_out)
        output_value, was_saturated = saturate(rounded, W_ACT)

        channel_outputs.append(output_value)
        saturated_channels += was_saturated

    return channel_outputs, saturated_channels


def main():
    parser = argparse.ArgumentParser(description="Modello di riferimento dell'encoder ECG")
    parser.add_argument("--stim", default="stim_10x12.mem", help="file dello stimolo")
    parser.add_argument("--weights", default="enc_w.mem", help="file dei pesi")
    parser.add_argument("--bias", default="enc_b.mem", help="file dei bias")
    parser.add_argument("--out", default="expected_10x128.mem", help="file dei valori attesi")
    parser.add_argument("--f-wenc", type=int, default=F_WENC_DEFAULT,
                        help="parametro F_WENC con cui e' istanziato il modulo")
    args = parser.parse_args()

    shift_out = args.f_wenc - F_ACT              # SH_OUT del modulo
    if shift_out < 0:
        raise ValueError(f"F_WENC ({args.f_wenc}) < F_ACT ({F_ACT}): l'RTL si ferma con errore")

    timesteps = load_stimulus(SCRIPT_DIR / args.stim)
    weights = load_weights(SCRIPT_DIR / args.weights)
    biases = load_biases(SCRIPT_DIR / args.bias)

    output_mask = (1 << W_ACT) - 1               # per scrivere i negativi in complemento a due
    file_lines = [
        f"// valori attesi di s4d_encoder_ecg_axis (generato da {Path(__file__).name})",
        f"// stimolo {args.stim}, pesi {args.weights}, bias {args.bias}, "
        f"F_WENC {args.f_wenc} (SH_OUT {shift_out})",
    ]
    total_saturated = 0

    for timestep_index, lead_samples in enumerate(timesteps):
        channel_outputs, saturated_channels = encode_timestep(
            lead_samples, weights, biases, shift_out)
        total_saturated += saturated_channels

        for channel, output_value in enumerate(channel_outputs):
            file_lines.append(f"{output_value & output_mask:04X}  "
                              f"// t={timestep_index} ch={channel} ({output_value})")

    # sat_sticky dell'RTL vale 1 se almeno un canale ha saturato
    file_lines.insert(2, f"// sat_sticky atteso: {1 if total_saturated else 0} "
                         f"({total_saturated} valori saturati)")

    expected_path = SCRIPT_DIR / args.out
    # fine riga LF, uguale su Windows e su Linux
    with open(expected_path, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")

    print(f"{expected_path.name}: {len(timesteps) * H} valori "
          f"({len(timesteps)} passi x {H} canali), F_WENC {args.f_wenc}, SH_OUT {shift_out}")
    print(f"valori saturati: {total_saturated}  ->  sat_sticky atteso = "
          f"{1 if total_saturated else 0}")


if __name__ == "__main__":
    main()
