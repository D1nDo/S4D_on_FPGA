"""
gen_encoder_stimulus.py -- stimolo riconoscibile per il testbench di s4d_encoder_ecg_axis

Genera stim_10x12.mem: 10 passi temporali (finestre) da 12 derivazioni ciascuno.
Ogni campione porta scritto nelle sue cifre esadecimali da dove viene:

    campione(passo t, derivazione c) = 0xTC00      T = t + 1  (1..A)
                                                   C = c      (0..B)

Sono numeri grandi di proposito: il modulo divide per 2^20 e un peso vale al
massimo 2^17, quindi campioni piccoli sparirebbero dopo lo shift.
Da T = 8 in su (passi 7, 8, 9) il campione a 16 bit con segno e' NEGATIVO
(0x8000 = -32768): cosi' si provano anche i numeri negativi.

Formato del file: UNA PAROLA AXI DA 32 BIT PER RIGA, esattamente come viaggia
su s_axis_tdata (vedi intestazione di s4d_encoder_ecg_axis.v):

    bit [15:0]  = campione pari    (c0, c2, c4, ...)
    bit [31:16] = campione dispari (c1, c3, c5, ...)

Quindi 6 righe per passo temporale e 60 righe in tutto. Il testbench le legge
con $readmemh e le mette sul bus cosi' come sono, senza reimpacchettarle.
Le righe che iniziano con // sono commenti: $readmemh li ignora.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Configurazione
# ----------------------------------------------------------------------------
NUM_TIMESTEPS = 10        # numero di finestre (passi temporali) da generare
NUM_LEADS = 12           # derivazioni per passo temporale (parametro NIN del modulo)
SAMPLE_BITS = 16         # larghezza di un campione (W_X nel modulo, formato Q5.11)
SAMPLES_PER_WORD = 2     # campioni impacchettati in una parola AXI da 32 bit

# maschera per tenere solo i 16 bit del campione (serve se un giorno si usano valori negativi)
SAMPLE_MASK = (1 << SAMPLE_BITS) - 1

# il file viene scritto accanto a questo script, nella cartella del testbench
OUTPUT_FILE = Path(__file__).with_name("stim_10x12.mem")


def pack_axi_word(even_lead_sample, odd_lead_sample):
    """Unisce due campioni da 16 bit in una parola da 32 bit.

    Il campione della derivazione pari va nei 16 bit bassi, quello della
    derivazione dispari nei 16 bit alti: e' l'ordine che il modulo si aspetta.
    """
    low_half = even_lead_sample & SAMPLE_MASK                    # bit [15:0]
    high_half = (odd_lead_sample & SAMPLE_MASK) << SAMPLE_BITS   # bit [31:16]
    return high_half | low_half


def main():
    file_lines = []

    for timestep in range(NUM_TIMESTEPS):
        # i 12 campioni di questo passo temporale, nella forma 0xTC00:
        #   T = passo + 1 nella cifra esadecimale piu' alta (bit 15..12)
        #   C = numero della derivazione nella seconda cifra (bit 11..8)
        # da T = 8 in su il bit 15 e' a 1, quindi il campione a 16 bit con segno e' negativo
        lead_samples = [((timestep + 1) << 12) | (lead << 8) for lead in range(NUM_LEADS)]

        # riga di commento che separa le finestre nel file
        file_lines.append(
            f"// passo temporale {timestep}: campioni 0x{lead_samples[0]:04X}..0x{lead_samples[-1]:04X}"
        )

        # si scorrono le derivazioni a coppie: (c0,c1), (c2,c3), ..., (c10,c11)
        for even_lead in range(0, NUM_LEADS, SAMPLES_PER_WORD):
            odd_lead = even_lead + 1
            axi_word = pack_axi_word(lead_samples[even_lead], lead_samples[odd_lead])

            # 8 cifre esadecimali = 32 bit; il commento dice quali campioni contiene la parola
            file_lines.append(
                f"{axi_word:08X}  // c{odd_lead}=0x{lead_samples[odd_lead]:04X}"
                f" c{even_lead}=0x{lead_samples[even_lead]:04X}"
            )

    # fine riga LF, uguale su Windows e su Linux
    with open(OUTPUT_FILE, "w", newline="\n") as mem_file:
        mem_file.write("\n".join(file_lines) + "\n")

    total_words = NUM_TIMESTEPS * NUM_LEADS // SAMPLES_PER_WORD
    print(f"{OUTPUT_FILE.name}: {total_words} parole da 32 bit "
          f"({NUM_TIMESTEPS} passi x {NUM_LEADS} derivazioni)")


if __name__ == "__main__":
    main()
