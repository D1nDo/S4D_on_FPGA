"""
gen_core_stimulus.py -- stimolo per il testbench di s4d_core_axis

Genera stim_2x8x128.mem: 2 sequenze ("immagini"), ognuna di 8 passi temporali
da 128 canali (1024 campioni per sequenza, 2048 in tutto). Un campione a 16 bit
per riga, nell'ordine in cui entra nel core: passo 0 canali 0..127, passo 1
canali 0..127, ... L'ultimo campione di ogni sequenza e' quello su cui il
testbench alza tlast.

I campioni sono numeri casuali: con i pesi veri della rete non esiste un
ingresso "leggibile", conta solo che l'uscita sia uguale a quella del modello.
Le due sequenze hanno ampiezze diverse:

    sequenza A   campioni tra -2048 e +2047  (da -1 a +1 nel formato Q5.11)
                 ampiezza moderata: la rete lavora lontano dai limiti
    sequenza B   campioni su tutta la scala a 16 bit
                 serve a far lavorare saturazioni e casi limite

Due sequenze diverse una dopo l'altra provano anche che il core riparte da zero
a ogni sequenza: gli stati dei filtri non devono passare dalla prima alla seconda.

I numeri casuali vengono da una formula scritta qui sotto, quindi il file esce
identico su qualsiasi computer.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Dimensioni (L e' ridotto a 8 per la prova: vedi intestazione del testbench)
# ----------------------------------------------------------------------------
H = 128                       # canali per passo temporale
L = 8                         # passi temporali per sequenza
SAMPLE_BITS = 16              # bit di un campione (W_ACT nel modulo)

SAMPLE_MASK = (1 << SAMPLE_BITS) - 1
SAMPLE_HIGHEST = (1 << (SAMPLE_BITS - 1)) - 1      # +32767

OUTPUT_FILE = Path(__file__).with_name("stim_2x8x128.mem")

# ----------------------------------------------------------------------------
# Generatore di numeri casuali scritto per esteso (riproducibile)
# ----------------------------------------------------------------------------
RANDOM_MULTIPLIER = 1103515245
RANDOM_INCREMENT = 12345
RANDOM_MODULUS = 1 << 31


def random_sequence(random_start, amplitude_bits):
    """Sequenza di campioni casuali con segno su amplitude_bits bit
    (12 bit -> da -2048 a +2047, 16 bit -> tutta la scala)."""
    generator_state = random_start
    samples = []
    for timestep in range(L):
        timestep_samples = []
        for channel in range(H):
            generator_state = (RANDOM_MULTIPLIER * generator_state + RANDOM_INCREMENT) % RANDOM_MODULUS
            raw_16_bits = (generator_state >> 15) & SAMPLE_MASK     # si usano i bit alti, i piu' casuali
            full_scale_sample = raw_16_bits - (1 << SAMPLE_BITS) if raw_16_bits > SAMPLE_HIGHEST else raw_16_bits
            # si riduce l'ampiezza scartando i bit bassi (lo spostamento a destra conserva il segno)
            timestep_samples.append(full_scale_sample >> (SAMPLE_BITS - amplitude_bits))
        samples.append(timestep_samples)
    return samples


def main():
    sequences = [
        ("A", "campioni casuali di ampiezza moderata (da -2048 a +2047)", random_sequence(2024, 12)),
        ("B", "campioni casuali su tutta la scala a 16 bit", random_sequence(7777, 16)),
    ]

    file_lines = [f"// stimolo di s4d_core_axis: {len(sequences)} sequenze x {L} passi x {H} canali"
                  f"  (generato da {Path(__file__).name})"]

    for sequence_name, description, samples in sequences:
        file_lines.append(f"// ===== sequenza {sequence_name}: {description} =====")
        for timestep in range(L):
            file_lines.append(f"// sequenza {sequence_name}, passo {timestep}")
            for channel in range(H):
                sample = samples[timestep][channel]
                # 4 cifre esadecimali = 16 bit in complemento a due; nel commento il valore con segno
                file_lines.append(f"{sample & SAMPLE_MASK:04X}  // {sequence_name} t={timestep} ch={channel} ({sample})")

    # fine riga LF, uguale su Windows e su Linux
    with open(OUTPUT_FILE, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")

    print(f"{OUTPUT_FILE.name}: {len(sequences) * L * H} campioni "
          f"({len(sequences)} sequenze x {L} passi x {H} canali)")


if __name__ == "__main__":
    main()
