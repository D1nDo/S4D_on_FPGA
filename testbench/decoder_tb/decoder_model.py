"""
decoder_model.py -- modello di riferimento di s4d_decoder_axis

Rifa' in Python, con numeri interi, gli stessi calcoli del modulo Verilog e
scrive i pacchetti che il modulo DEVE produrre. Il testbench confronta l'uscita
dell'RTL con questo file, parola per parola.

Calcolo, per ogni sequenza di L passi temporali da H canali:

  1. s4d_affine      a[t][ch] = satura16( arrotonda( (w[ch]*x[t][ch] + b[ch]) / 2^BN_SH ) )
  2. s4d_meanpool    m[ch]    = satura16( arrotonda( somma su t di a[t][ch] / 2^LOG2_L ) )
  3. s4d_classifier  xq[ch]   = satura8 ( arrotonda( m[ch] / 2^RQ_SHIFT ) )
                     logit[k] = b_cls[k] + somma su ch di W_cls[ch][k] * xq[ch]
                     classe   = indice del logit piu' grande (a parita' vince il primo)
  4. s4d_decoder_axis  pacchetto di K + 1 parole da 32 bit:
                     K logit, poi la parola di classe  0xC1 | flag di errore | classe

Il modello legge GLI STESSI file che legge l'RTL (stimolo, l6_bn_w, l6_bn_b,
cls_w, cls_b), quindi verifica anche che il formato dei .mem sia interpretato
allo stesso modo.

File scritto: una parola da 32 bit per riga (8 cifre esadecimali), nell'ordine
in cui il modulo le emette: per ogni sequenza i K logit e poi la parola di classe.

Uso (i valori tra parentesi quadre sono quelli di default):
    python3 decoder_model.py [--stim stim_4x8x128.mem] [--out expected_4x6.mem]
"""

import argparse
from pathlib import Path

# ----------------------------------------------------------------------------
# Parametri del modulo (stessi nomi di s4d_decoder_axis.v)
#   K, LOG2_K, BN_SH : valori del progetto ECG (impostati nel block design)
#   L, LOG2_L        : RIDOTTI per la prova (in scheda L = 1024, LOG2_L = 10)
# ----------------------------------------------------------------------------
H = 128              # canali
L = 8                # passi temporali per sequenza (prova)
LOG2_L = 3           # la media divide per 2^LOG2_L = L
K = 5                # classi
LOG2_K = 3           # bit dell'indice di classe

W_ACT = 16           # bit di un campione in ingresso e dopo normalizzazione e media
W_CIN = 8            # bit dell'ingresso del classificatore
W_CLS = 8            # bit di un peso del classificatore
W_ACC = 32           # bit dell'accumulatore della media e dei logit
RQ_SHIFT = 6         # il classificatore divide la media per 2^6 = 64

BN_SH = 13           # shift della normalizzazione
BN_W_W = 16          # bit di un guadagno della normalizzazione
BN_W_B = 32          # bit di un bias della normalizzazione
BN_W_ACC = 34        # bit dell'accumulatore della normalizzazione

MAGIC = 0xC1         # byte fisso in testa alla parola di classe

SCRIPT_DIR = Path(__file__).parent   # cartella in cui si cercano i file .mem


# ----------------------------------------------------------------------------
# Funzioni di aritmetica a larghezza fissa
# ----------------------------------------------------------------------------
def to_signed(raw_value, num_bits):
    """Interpreta i num_bits meno significativi di raw_value come numero con segno
    in complemento a due (e' quello che fa Verilog con un 'wire signed')."""
    raw_value &= (1 << num_bits) - 1
    sign_bit = 1 << (num_bits - 1)
    return raw_value - (1 << num_bits) if raw_value & sign_bit else raw_value


def round_shift_right(value, shift):
    """Divide per 2^shift arrotondando al piu' vicino (le meta' vanno verso l'alto):
    (value + 2^(shift-1)) >> shift, come '(x + RND) >>> SH' nell'RTL."""
    if shift == 0:
        return value
    return (value + (1 << (shift - 1))) >> shift


def saturate(value, num_bits):
    """Limita value all'intervallo di un numero con segno a num_bits bit.
    Restituisce anche True se il valore e' stato tagliato."""
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
        raise ValueError(f"{stimulus_path.name}: {len(samples)} campioni, "
                         f"non multiplo di {samples_per_sequence}")
    sequences = []
    for sequence_start in range(0, len(samples), samples_per_sequence):
        sequence = [samples[sequence_start + timestep * H: sequence_start + (timestep + 1) * H]
                    for timestep in range(L)]
        sequences.append(sequence)
    return sequences


def load_classifier_weights(weight_path):
    """Restituisce weights[ch][k]: il peso della classe k sta nei bit [k*8 +: 8] della riga."""
    return [[to_signed(row_bits >> (class_index * W_CLS), W_CLS) for class_index in range(K)]
            for row_bits in read_mem_file(weight_path, H)]


# ----------------------------------------------------------------------------
# I calcoli del decoder, un blocco alla volta
# ----------------------------------------------------------------------------
def affine(sample, weight, bias):
    """s4d_affine: prodotto, bias, arrotondamento, shift e saturazione a 16 bit."""
    accumulator = to_signed(weight * sample + bias, BN_W_ACC)          # accumulatore a 34 bit
    rounded = round_shift_right(accumulator, BN_SH)                    # (acc + RND) >>> BN_SH
    return saturate(rounded, W_ACT)


def decode_sequence(sequence, affine_weights, affine_biases, classifier_weights, classifier_biases):
    """Calcola logit e classe di una sequenza. Restituisce anche quante saturazioni ci sono state."""
    saturation_counts = {"normalizzazione": 0, "media": 0, "riduzione a 8 bit": 0}

    # 1 + 2. normalizzazione di ogni campione e somma nel tempo, canale per canale
    channel_sums = [0] * H
    for timestep_samples in sequence:
        for channel in range(H):
            normalized, was_saturated = affine(timestep_samples[channel],
                                               affine_weights[channel], affine_biases[channel])
            saturation_counts["normalizzazione"] += was_saturated
            channel_sums[channel] = to_signed(channel_sums[channel] + normalized, W_ACC)   # accumulatore a 32 bit

    # 2. media: divisione per 2^LOG2_L con arrotondamento, saturazione a 16 bit
    channel_means = []
    for channel in range(H):
        mean_value, was_saturated = saturate(round_shift_right(channel_sums[channel], LOG2_L), W_ACT)
        saturation_counts["media"] += was_saturated
        channel_means.append(mean_value)

    # 3a. riduzione a 8 bit dell'ingresso del classificatore
    classifier_inputs = []
    for channel in range(H):
        reduced, was_saturated = saturate(round_shift_right(channel_means[channel], RQ_SHIFT), W_CIN)
        saturation_counts["riduzione a 8 bit"] += was_saturated
        classifier_inputs.append(reduced)

    # 3b. logit: si parte dal bias e si aggiunge un prodotto per canale (accumulatore a 32 bit)
    logits = []
    for class_index in range(K):
        accumulator = classifier_biases[class_index]
        for channel in range(H):
            accumulator = to_signed(accumulator + classifier_weights[channel][class_index]
                                    * classifier_inputs[channel], W_ACC)
        logits.append(accumulator)

    # 3c. classe: si scorre dal logit 0 e si tiene il piu' grande; a parita' resta il primo
    best_logit = -(1 << (W_ACC - 1))      # valore iniziale dell'RTL: il piu' negativo
    best_class = 0
    for class_index, logit in enumerate(logits):
        if logit > best_logit:
            best_logit, best_class = logit, class_index

    return logits, best_class, saturation_counts


def class_word(class_index, error_overrun=0, error_length=0):
    """Parola di classe del pacchetto: [31:24] MAGIC, [17] ERR_OVR, [16] ERR_LEN, [LOG2_K-1:0] classe."""
    return (MAGIC << 24) | (error_overrun << 17) | (error_length << 16) | class_index


def main():
    parser = argparse.ArgumentParser(description="Modello di riferimento del decoder")
    parser.add_argument("--stim", default="stim_4x8x128.mem", help="file dello stimolo")
    parser.add_argument("--bn-weights", default="l6_bn_w.mem", help="guadagni della normalizzazione")
    parser.add_argument("--bn-bias", default="l6_bn_b.mem", help="bias della normalizzazione")
    parser.add_argument("--cls-weights", default="cls_w.mem", help="pesi del classificatore")
    parser.add_argument("--cls-bias", default="cls_b.mem", help="bias del classificatore")
    parser.add_argument("--out", default="expected_4x6.mem", help="file dei pacchetti attesi")
    args = parser.parse_args()

    sequences = load_stimulus(SCRIPT_DIR / args.stim)
    affine_weights = [to_signed(raw, BN_W_W) for raw in read_mem_file(SCRIPT_DIR / args.bn_weights, H)]
    affine_biases = [to_signed(raw, BN_W_B) for raw in read_mem_file(SCRIPT_DIR / args.bn_bias, H)]
    classifier_weights = load_classifier_weights(SCRIPT_DIR / args.cls_weights)
    classifier_biases = [to_signed(raw, W_ACC) for raw in read_mem_file(SCRIPT_DIR / args.cls_bias, K)]

    word_mask = (1 << 32) - 1
    file_lines = [
        f"// pacchetti attesi di s4d_decoder_axis (generato da {Path(__file__).name})",
        f"// stimolo {args.stim}, L {L}, K {K}, BN_SH {BN_SH}: per ogni sequenza {K} logit + 1 parola di classe",
    ]

    for sequence_index, sequence in enumerate(sequences):
        logits, best_class, saturation_counts = decode_sequence(
            sequence, affine_weights, affine_biases, classifier_weights, classifier_biases)

        for class_index, logit in enumerate(logits):
            file_lines.append(f"{logit & word_mask:08X}  // sequenza {sequence_index}: logit {class_index} = {logit}")
        # i flag di errore attesi sono 0: lo stimolo rispetta la lunghezza e il testbench svuota in tempo
        file_lines.append(f"{class_word(best_class):08X}  // sequenza {sequence_index}: classe {best_class}")

        saturation_text = ", ".join(f"{stage} {count}" for stage, count in saturation_counts.items())
        print(f"sequenza {sequence_index}: logit {logits} -> classe {best_class}   (saturazioni: {saturation_text})")

    # fine riga LF, uguale su Windows e su Linux
    with open(SCRIPT_DIR / args.out, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")

    print(f"{args.out}: {len(sequences) * (K + 1)} parole ({len(sequences)} pacchetti da {K + 1})")


if __name__ == "__main__":
    main()
