/* ============================================================================
 * s4d_PTB_XL_2158ECG.c -- test da userspace Linux (via /dev/mem) per il
 * bitstream S4D_PTB_XL_UltraTiny (ECG PTB-XL, 12 derivazioni, 5 superclassi).
 *
 * Rispetto alla versione precedente cambia il PROTOCOLLO DI VALUTAZIONE:
 * quello standard di PTB-XL "superdiagnostic" -- test set = fold 10,
 * etichette MULTI-LABEL, metrica principale macro AUC one-vs-rest
 * (term-centric: una AUC per superclasse, poi media non pesata).
 * L'accuratezza top-1 resta stampata solo come riferimento.
 *
 * L'AUC si calcola dai LOGIT interi che il decoder gia' manda (K beat da
 * 32 bit): l'AUC dipende solo dall'ordinamento dei punteggi, quindi non
 * serve nessuna softmax ne' aritmetica in virgola mobile sul percorso dati.
 * Formula di Mann-Whitney con midrank sui pareggi:
 *     AUC = (somma dei ranghi dei positivi - n_pos(n_pos+1)/2) / (n_pos*n_neg)
 *
 * Mappa DMA (da s4d_PTB_XL.bd):
 *   axi_dma_0 @ 0x40400000 -- PESI: MM2S (64 bit) -> s4d_wrom_axis
 *   axi_dma_1 @ 0x40410000 -- DATI: MM2S (32 bit) -> s4d_encoder_ecg_axis
 *                              S2MM (32 bit) <- s4d_decoder_axis
 *
 * Campione: L=1024 timestep x 12 derivazioni, int16 Q5.11 little-endian,
 * ordine [t][c] -> 24576 byte. Uscita: 5 logit int32 + 1 beat di classe.
 *
 * File attesi nella working dir:
 *   w_stream_ECG.bin     -- 718896 byte, pesi dei 6 layer
 *   samples_2158ECG.bin  -- 2158 * 24576 byte
 *   labels_2158ECG.bin   -- 2158 byte, MASCHERA di bit:
 *                           bit0 CD, bit1 HYP, bit2 MI, bit3 NORM, bit4 STTC
 *
 * Prima di lanciarlo: impostare FCLK0 con fclk_set (la scheda parte a 125 MHz).
 *
 * Uso:  ./s4d_PTB_XL_2158ECG [N]      (default N = 2158)
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <time.h>

#define DMA_TIMEOUT_SEC  300.0

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ------------------------------- CONFIG --------------------------------- */

#define BUILD_TAG  "ECG BUILD 2 -- PTB-XL superdiagnostic, macro AUC one-vs-rest"

#define DMA0_BASE_PHYS   0x40400000UL   /* axi_dma_0: path pesi */
#define DMA1_BASE_PHYS   0x40410000UL   /* axi_dma_1: path dati */
#define DMA_REG_SPAN     0x10000UL

#define NLAYER           6
#define NBEAT            14977
#define AXI_DW_BYTES     8
#define LAYER_BYTES      (NBEAT * AXI_DW_BYTES)          /* 119816 */
#define WEIGHTS_TOTAL_BYTES (LAYER_BYTES * NLAYER)       /* 718896 */

#define L_SEQ            1024
#define N_LEADS          12
#define SAMPLE_BYTES     (L_SEQ * N_LEADS * 2)           /* 24576 */
#define K_CLASSES        5
#define OUTPUT_BYTES     ((K_CLASSES + 1) * 4)           /* 24 */

#define N_MAX            2158

static const char *CLASS_NAME[K_CLASSES] = { "CD", "HYP", "MI", "NORM", "STTC" };

#define DDR_RESERVED_BASE   0x1D000000UL
#define WEIGHTS_PHYS   (DDR_RESERVED_BASE)
#define SAMPLE_PHYS    (WEIGHTS_PHYS + WEIGHTS_TOTAL_BYTES)
#define OUTPUT_PHYS    (SAMPLE_PHYS + SAMPLE_BYTES)
#define DDR_SPAN       (WEIGHTS_TOTAL_BYTES + SAMPLE_BYTES + OUTPUT_BYTES)

#define WEIGHTS_BIN_PATH  "w_stream_ECG.bin"
#define SAMPLES_BIN_PATH  "samples_2158ECG.bin"
#define LABELS_BIN_PATH   "labels_2158ECG.bin"

/* --------------------------- Registri AXI DMA ---------------------------- */

#define REG_MM2S_DMACR   0x00
#define REG_MM2S_DMASR   0x04
#define REG_MM2S_SA      0x18
#define REG_MM2S_LENGTH  0x28

#define REG_S2MM_DMACR   0x30
#define REG_S2MM_DMASR   0x34
#define REG_S2MM_DA      0x48
#define REG_S2MM_LENGTH  0x58

#define DMACR_RS         (1u << 0)
#define DMACR_RESET      (1u << 2)
#define DMASR_IDLE       (1u << 1)
#define DMASR_ERR_MASK   0x770u
#define DMASR_IOC_IRQ    (1u << 12)
#define DMASR_IRQ_MASK   0x7000u

static volatile uint32_t *map_regs(int fd, uint64_t phys, size_t span)
{
    void *p = mmap(NULL, span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)phys);
    if (p == MAP_FAILED) {
        fprintf(stderr, "mmap regs @0x%lx: %s\n", (unsigned long)phys, strerror(errno));
        exit(1);
    }
    return (volatile uint32_t *)p;
}

static inline void reg_w(volatile uint32_t *base, unsigned off, uint32_t val)
{
    base[off / 4] = val;
}
static inline uint32_t reg_r(volatile uint32_t *base, unsigned off)
{
    return base[off / 4];
}

static void dma_reset(volatile uint32_t *base, unsigned dmacr_off)
{
    reg_w(base, dmacr_off, DMACR_RESET);
    int guard = 100000;
    while ((reg_r(base, dmacr_off) & DMACR_RESET) && guard--) {}
    if (guard <= 0) {
        fprintf(stderr, "dma_reset: timeout\n");
        exit(1);
    }
}

static void dma_wait(volatile uint32_t *base, unsigned sr_off, const char *tag,
                     uint64_t phys, uint32_t length)
{
    double t0 = now_sec();
    double last_print = t0;
    for (;;) {
        uint32_t sr = reg_r(base, sr_off);
        if (sr & DMASR_ERR_MASK) {
            fprintf(stderr, "%s errore, DMASR=0x%08x (addr=0x%lx len=%u)\n",
                    tag, sr, (unsigned long)phys, length);
            exit(1);
        }
        if (sr & (DMASR_IDLE | DMASR_IOC_IRQ)) {
            reg_w(base, sr_off, sr & DMASR_IRQ_MASK);
            return;
        }
        double now = now_sec();
        if (now - last_print > 10.0) {
            printf("    ...%s ancora in corso dopo %.0fs (DMASR=0x%08x)\n", tag, now - t0, sr);
            fflush(stdout);
            last_print = now;
        }
        if (now - t0 > DMA_TIMEOUT_SEC) {
            fprintf(stderr, "%s timeout dopo %.1fs, DMASR=0x%08x (addr=0x%lx len=%u)\n",
                    tag, DMA_TIMEOUT_SEC, sr, (unsigned long)phys, length);
            exit(1);
        }
    }
}

static void dma_mm2s(volatile uint32_t *base, uint64_t phys_src, uint32_t length)
{
    reg_w(base, REG_MM2S_DMACR, DMACR_RS);
    reg_w(base, REG_MM2S_SA, (uint32_t)phys_src);
    reg_w(base, REG_MM2S_LENGTH, length);
    dma_wait(base, REG_MM2S_DMASR, "MM2S", phys_src, length);
}

static void dma_s2mm_start(volatile uint32_t *base, uint64_t phys_dst, uint32_t length)
{
    reg_w(base, REG_S2MM_DMACR, DMACR_RS);
    reg_w(base, REG_S2MM_DA, (uint32_t)phys_dst);
    reg_w(base, REG_S2MM_LENGTH, length);
}

static void load_all_layers(volatile uint32_t *dma_w)
{
    for (int layer = 0; layer < NLAYER; layer++)
        dma_mm2s(dma_w, WEIGHTS_PHYS + (uint64_t)layer * LAYER_BYTES, LAYER_BYTES);
}

static void read_exact(const char *path, FILE *f, void *dst, size_t n)
{
    if (fread(dst, 1, n, f) != n) {
        fprintf(stderr, "%s troppo corto (attesi altri %zu byte)\n", path, n);
        exit(1);
    }
}

/* ------------------------------ metriche -------------------------------- */

static int32_t scores[N_MAX][K_CLASSES];
static uint8_t truth[N_MAX];

typedef struct { int32_t score; uint8_t pos; } entry_t;

static int cmp_entry(const void *a, const void *b)
{
    int32_t x = ((const entry_t *)a)->score, y = ((const entry_t *)b)->score;
    return (x > y) - (x < y);
}

/* AUC one-vs-rest con la statistica di Mann-Whitney, midrank sui pareggi.
 * Ritorna -1 se la classe non ha sia positivi sia negativi. */
static double auc_one_class(int n, int k, entry_t *buf)
{
    long n_pos = 0;
    for (int i = 0; i < n; i++) {
        buf[i].score = scores[i][k];
        buf[i].pos   = (truth[i] >> k) & 1u;
        n_pos += buf[i].pos;
    }
    long n_neg = n - n_pos;
    if (n_pos == 0 || n_neg == 0) return -1.0;

    qsort(buf, (size_t)n, sizeof(entry_t), cmp_entry);

    /* somma dei ranghi dei positivi, ranghi da 1, pareggi a rango medio */
    double rank_sum = 0.0;
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && buf[j + 1].score == buf[i].score) j++;
        double midrank = (double)(i + 1 + j + 1) / 2.0;   /* ranghi 1-based */
        for (int t = i; t <= j; t++)
            if (buf[t].pos) rank_sum += midrank;
        i = j + 1;
    }
    return (rank_sum - (double)n_pos * (n_pos + 1) / 2.0) / ((double)n_pos * n_neg);
}

int main(int argc, char **argv)
{
    int n_samples = N_MAX;
    if (argc > 1) {
        n_samples = atoi(argv[1]);
        if (n_samples < 2 || n_samples > N_MAX) {
            fprintf(stderr, "N deve essere fra 2 e %d\n", N_MAX);
            return 1;
        }
    }

    printf("=== %s ===\n", BUILD_TAG);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "apri /dev/mem (serve root): %s\n", strerror(errno));
        return 1;
    }

    volatile uint32_t *dma_w = map_regs(fd, DMA0_BASE_PHYS, DMA_REG_SPAN);
    volatile uint32_t *dma_d = map_regs(fd, DMA1_BASE_PHYS, DMA_REG_SPAN);

    void *ddr = mmap(NULL, DDR_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, (off_t)DDR_RESERVED_BASE);
    if (ddr == MAP_FAILED) {
        fprintf(stderr, "mmap DDR riservata @0x%lx (%d byte): %s\n",
                (unsigned long)DDR_RESERVED_BASE, DDR_SPAN, strerror(errno));
        return 1;
    }
    uint8_t *ddr_weights = (uint8_t *)ddr;
    uint8_t *ddr_sample  = ddr_weights + WEIGHTS_TOTAL_BYTES;
    volatile uint32_t *ddr_output = (volatile uint32_t *)(ddr_sample + SAMPLE_BYTES);

    FILE *fw = fopen(WEIGHTS_BIN_PATH, "rb");
    if (!fw) { fprintf(stderr, "apri %s: %s\n", WEIGHTS_BIN_PATH, strerror(errno)); return 1; }
    read_exact(WEIGHTS_BIN_PATH, fw, ddr_weights, WEIGHTS_TOTAL_BYTES);
    fclose(fw);

    FILE *fl = fopen(LABELS_BIN_PATH, "rb");
    if (!fl) { fprintf(stderr, "apri %s: %s\n", LABELS_BIN_PATH, strerror(errno)); return 1; }
    read_exact(LABELS_BIN_PATH, fl, truth, (size_t)n_samples);
    fclose(fl);

    FILE *fs = fopen(SAMPLES_BIN_PATH, "rb");
    if (!fs) { fprintf(stderr, "apri %s: %s\n", SAMPLES_BIN_PATH, strerror(errno)); return 1; }

    dma_reset(dma_w, REG_MM2S_DMACR);
    dma_reset(dma_d, REG_MM2S_DMACR);
    dma_reset(dma_d, REG_S2MM_DMACR);

    int top1_hit = 0, magic_errors = 0, len_errors = 0, ovr_errors = 0;

    printf("Eseguo %d ECG (%d byte ciascuno, pesi ricaricati ogni volta)...\n\n",
           n_samples, SAMPLE_BYTES);

    double t_start = now_sec();

    for (int i = 0; i < n_samples; i++) {
        read_exact(SAMPLES_BIN_PATH, fs, ddr_sample, SAMPLE_BYTES);
        memset((void *)ddr_output, 0, OUTPUT_BYTES);

        dma_s2mm_start(dma_d, OUTPUT_PHYS, OUTPUT_BYTES);

        /* ORDINE OBBLIGATO: campione prima dei pesi (vedi FSM del core). */
        dma_mm2s(dma_d, SAMPLE_PHYS, SAMPLE_BYTES);
        load_all_layers(dma_w);

        dma_wait(dma_d, REG_S2MM_DMASR, "S2MM", OUTPUT_PHYS, OUTPUT_BYTES);

        for (int k = 0; k < K_CLASSES; k++)
            scores[i][k] = (int32_t)ddr_output[k];

        uint32_t class_beat = ddr_output[K_CLASSES];
        uint8_t  magic     = (class_beat >> 24) & 0xFF;
        int      err_ovr   = (class_beat >> 17) & 0x1;
        int      err_len   = (class_beat >> 16) & 0x1;
        int      predicted = class_beat & 0x7;

        if (magic != 0xC1) magic_errors++;
        if (err_len) len_errors++;
        if (err_ovr) ovr_errors++;
        /* top-1: conta come corretta se la classe predetta e' fra quelle vere */
        if (predicted < K_CLASSES && ((truth[i] >> predicted) & 1u)) top1_hit++;

        if ((i + 1) % 100 == 0 || i + 1 == n_samples) {
            double el = now_sec() - t_start;
            printf("  %4d/%d  top1 %.1f%%  %.3f s/campione\n",
                   i + 1, n_samples, 100.0 * top1_hit / (i + 1), el / (i + 1));
            fflush(stdout);
        }
    }
    fclose(fs);

    double elapsed = now_sec() - t_start;

    /* ------------------------------ risultati --------------------------- */
    entry_t *buf = malloc(sizeof(entry_t) * (size_t)n_samples);
    if (!buf) { fprintf(stderr, "malloc\n"); return 1; }

    printf("\n=== PTB-XL superdiagnostic, fold 10 ===\n");
    printf("%-6s %9s %9s %8s\n", "classe", "positivi", "negativi", "AUC");

    double sum_auc = 0.0;
    int n_valid = 0;
    for (int k = 0; k < K_CLASSES; k++) {
        long pos = 0;
        for (int i = 0; i < n_samples; i++) pos += (truth[i] >> k) & 1u;
        double a = auc_one_class(n_samples, k, buf);
        if (a < 0) {
            printf("%-6s %9ld %9ld %8s\n", CLASS_NAME[k], pos, n_samples - pos, "n/d");
        } else {
            printf("%-6s %9ld %9ld %8.4f\n", CLASS_NAME[k], pos, n_samples - pos, a);
            sum_auc += a;
            n_valid++;
        }
    }
    free(buf);

    printf("\nMACRO AUC (term-centric, media su %d classi): %.4f\n",
           n_valid, n_valid ? sum_auc / n_valid : 0.0);
    printf("Top-1 (riferimento, non standard): %d/%d = %.2f%%\n",
           top1_hit, n_samples, 100.0 * top1_hit / n_samples);
    printf("Tempo totale: %.1f s (%.3f s/campione, ricarica pesi inclusa)\n",
           elapsed, elapsed / n_samples);

    if (magic_errors || len_errors || ovr_errors) {
        printf("\nATTENZIONE: MAGIC errati=%d  ERR_LEN=%d  ERR_OVR=%d -- "
               "non fidarsi dei risultati sopra.\n",
               magic_errors, len_errors, ovr_errors);
    }

    munmap(ddr, DDR_SPAN);
    munmap((void *)dma_w, DMA_REG_SPAN);
    munmap((void *)dma_d, DMA_REG_SPAN);
    close(fd);
    return 0;
}
