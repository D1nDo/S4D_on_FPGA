"""
gen_decoder_stimulus.py -- stimolo per il testbench di s4d_decoder_axis

Genera stim_4x8x128.mem: 4 sequenze, ognuna di 8 passi temporali da 128 canali
(1024 campioni per sequenza, 4096 in tutto). Un campione a 16 bit per riga,
nell'ordine in cui entra nel modulo: passo 0 canali 0..127, passo 1 canali 0..127, ...
L'ultimo campione di ogni sequenza e' quello su cui il testbench alza tlast.

COSA FA IL DECODER (per ogni sequenza)
    1. normalizzazione   a[t][ch] = guadagno[ch] * x[t][ch] + offset[ch] + 0,5   (poi arrotonda)
    2. media nel tempo   m[ch]    = media di a[t][ch] sugli 8 passi             (poi arrotonda)
    3. riduzione a 8 bit xq[ch]   = m[ch] / 64                                  (poi arrotonda)
    4. classificatore    logit[k] = k + somma dei xq dei 25 canali che votano per k
    5. classe            = indice del logit piu' grande

COME SONO COSTRUITE LE SEQUENZE A, B, C (a ritroso, partendo dal risultato voluto)
    Si sceglie un "voto" V[k] per ogni classe. Ogni canale che vota per la
    classe k deve arrivare al classificatore con xq = V[k], quindi:
        logit[k] = k + 25 * V[k]
    I voti sono multipli di 4, cosi' 25 * V e' un multiplo di 100 e il logit
    si legge in decimale: centinaia = voto / 4, unita' = numero della classe.

    Per ogni passo si decide il valore che deve uscire dalla normalizzazione e
    da li' si torna indietro al campione da mandare:
        x[t][ch] = ( a_voluto - 1 - offset[ch] ) / guadagno[ch]
    (il "- 1" e' la mezza unita' del bias, che l'arrotondamento porta a 1).
    offset[ch] e guadagno[ch] sono quelli di bias_gen.py e weight_gen.py: la
    normalizzazione li deve togliere, canale per canale.

    Sequenze A e B -- lontane dalle soglie di arrotondamento
        a_voluto = 64 * V + onda[t] + 1
        L'onda cambia a ogni passo ma ha somma zero sugli 8 passi: la media la
        deve cancellare. Se il modulo saltasse o contasse due volte un passo,
        la media non tornerebbe.
        A: voti [-4,  8, 20,  4, 12] -> logit [-100, 201, 502, 103, 304] -> classe 2
        B: voti [-8, 16,  4, 12, 28] -> logit [-200, 401, 102, 303, 704] -> classe 4

    Sequenza C -- esattamente SULLE soglie di arrotondamento
        a_voluto = 64 * V - 33 per sei passi, 64 * V - 31 per due passi
        La media esatta vale 64 * V - 32,5: l'arrotondamento la porta a
        64 * V - 32, che diviso 64 fa V - 0,5 e viene arrotondato a V.
        I tre arrotondamenti (normalizzazione, media, riduzione a 8 bit) sono
        tutti al limite: se ne manca uno, ogni voto scende di 1 e ogni logit di 25.
        C: voti [-4, 12,  8, 24,  4] -> logit [-100, 301, 202, 603, 104] -> classe 3

SEQUENZA D
    Campioni casuali su tutta la scala a 16 bit. Non e' leggibile a occhio:
    serve a far lavorare le saturazioni, che A, B e C evitano di proposito.
    I numeri casuali vengono da una formula scritta qui sotto, quindi il file
    esce identico su qualsiasi computer.

Sequenze diverse una dopo l'altra provano anche che il modulo riparte da zero
a ogni sequenza, senza portarsi dietro le somme della precedente.
"""

from pathlib import Path

from bias_gen import AFFINE_ROUNDING_STEP    # 1: effetto della mezza unita' del bias dopo l'arrotondamento
from bias_gen import affine_offset           # offset aggiunto dalla normalizzazione al canale
from weight_gen import affine_gain           # guadagno della normalizzazione sul canale

# ----------------------------------------------------------------------------
# Dimensioni (L e' ridotto a 8 per la prova: vedi intestazione del testbench)
# ----------------------------------------------------------------------------
H = 128                       # canali per passo temporale
L = 8                         # passi temporali per sequenza
K = 5                         # classi
SAMPLE_BITS = 16              # bit di un campione (W_ACT nel modulo)
REQUANT_DIVISOR = 64          # il classificatore divide la media per 2^RQ_SHIFT = 64
NUM_VOTING_CHANNELS = 125     # i canali 125..127 non votano: il loro voto e' 0

SAMPLE_MASK = (1 << SAMPLE_BITS) - 1
SAMPLE_HIGHEST = (1 << (SAMPLE_BITS - 1)) - 1      # +32767
SAMPLE_LOWEST = -(1 << (SAMPLE_BITS - 1))          # -32768

OUTPUT_FILE = Path(__file__).with_name("stim_4x8x128.mem")

# ----------------------------------------------------------------------------
# Sequenze A e B: voto di ogni classe e onda nel tempo (somma zero sugli 8 passi)
# ----------------------------------------------------------------------------
VOTES_A = [-4, 8, 20, 4, 12]                                    # vince la classe 2
VOTES_B = [-8, 16, 4, 12, 28]                                   # vince la classe 4
WAVE_A = [2048, -2048, 2048, -2048, 2048, -2048, 2048, -2048]   # alternata
WAVE_B = [-1792, -1280, -768, -256, 256, 768, 1280, 1792]       # rampa

# ----------------------------------------------------------------------------
# Sequenza C: voti e scostamento di ogni passo dal valore 64 * V
#   sei passi a -33 e due a -31: somma -260, cioe' media -32,5 (a meta' tra due interi)
# ----------------------------------------------------------------------------
VOTES_C = [-4, 12, 8, 24, 4]                                    # vince la classe 3
EDGE_OFFSETS_C = [-33, -33, -33, -31, -33, -33, -33, -31]

# ----------------------------------------------------------------------------
# Sequenza D: generatore di numeri casuali scritto per esteso (riproducibile)
# ----------------------------------------------------------------------------
RANDOM_START = 12345          # valore iniziale del generatore
RANDOM_MULTIPLIER = 1103515245
RANDOM_INCREMENT = 12345
RANDOM_MODULUS = 1 << 31


def sample_for_affine_output(wanted_after_affine, channel):
    """Campione da mandare perche' la normalizzazione del canale dia esattamente
    wanted_after_affine. Si torna indietro: si toglie cio' che il bias aggiunge
    (offset intero + 1 per la mezza unita' arrotondata) e si divide per il guadagno."""
    numerator = wanted_after_affine - AFFINE_ROUNDING_STEP - affine_offset(channel)
    gain = affine_gain(channel)
    assert numerator % gain == 0, "la divisione per il guadagno deve essere esatta"
    sample = numerator // gain
    assert SAMPLE_LOWEST <= sample <= SAMPLE_HIGHEST, "il campione non sta in 16 bit"
    return sample


def designed_sequence(class_votes, offset_per_timestep):
    """Costruisce una sequenza a ritroso dal voto voluto per ogni classe.
    offset_per_timestep[t] = di quanto l'uscita della normalizzazione al passo t
    si scosta da 64 * voto. Restituisce samples[t][ch]."""
    samples = []
    for timestep in range(L):
        timestep_samples = []
        for channel in range(H):
            # valore che il canale deve portare al classificatore (0 per i canali che non votano)
            vote = class_votes[channel % K] if channel < NUM_VOTING_CHANNELS else 0

            # valore voluto all'uscita della normalizzazione in questo passo
            wanted_after_affine = REQUANT_DIVISOR * vote + offset_per_timestep[timestep]

            timestep_samples.append(sample_for_affine_output(wanted_after_affine, channel))
        samples.append(timestep_samples)
    return samples


def random_sequence():
    """Sequenza di campioni casuali su tutta la scala a 16 bit con segno."""
    generator_state = RANDOM_START
    samples = []
    for timestep in range(L):
        timestep_samples = []
        for channel in range(H):
            generator_state = (RANDOM_MULTIPLIER * generator_state + RANDOM_INCREMENT) % RANDOM_MODULUS
            raw_16_bits = (generator_state >> 15) & SAMPLE_MASK     # si usano i bit alti, i piu' casuali
            sample = raw_16_bits - (1 << SAMPLE_BITS) if raw_16_bits > SAMPLE_HIGHEST else raw_16_bits
            timestep_samples.append(sample)
        samples.append(timestep_samples)
    return samples


def main():
    # le onde di A e B devono avere somma zero, altrimenti la media non darebbe 64 * voto + 1
    assert sum(WAVE_A) == 0 and sum(WAVE_B) == 0
    # in A e B l'uscita voluta e' 64 * voto + onda + 1 (dispari: serve per i canali con guadagno 2)
    offsets_a = [wave_value + 1 for wave_value in WAVE_A]
    offsets_b = [wave_value + 1 for wave_value in WAVE_B]

    sequences = [
        ("A", "voti [-4, 8, 20, 4, 12], onda alternata: deve vincere la classe 2",
         designed_sequence(VOTES_A, offsets_a)),
        ("B", "voti [-8, 16, 4, 12, 28], onda a rampa: deve vincere la classe 4",
         designed_sequence(VOTES_B, offsets_b)),
        ("C", "voti [-4, 12, 8, 24, 4] sulle soglie di arrotondamento: deve vincere la classe 3",
         designed_sequence(VOTES_C, EDGE_OFFSETS_C)),
        ("D", "campioni casuali a fondo scala: saturazioni",
         random_sequence()),
    ]

    file_lines = [f"// stimolo di s4d_decoder_axis: {len(sequences)} sequenze x {L} passi x {H} canali"
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
