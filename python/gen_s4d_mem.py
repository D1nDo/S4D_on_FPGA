#!/usr/bin/env python3
"""
gen_s4d_mem.py -- coefficienti e modello golden di un layer S4D.

Fa il lavoro che in S4D_matematica_completa.tex appartiene al training e non
all'hardware: esponenziali complesse, ZOH, riduzione a biquad reale. L'FPGA
riceve solo quattro coefficienti costanti e due registri per coppia di poli.

  1. lambda = -exp(log_A_real) + i*A_imag,  Delta = exp(log_dt)
  2. ZOH:    lbar = exp(Delta*lambda),  Bbar = (lbar-1)/lambda
  3. biquad: a1 = -2Re(lbar), a2 = |lbar|^2,
             b0 = 2Re(C*Bbar), b1 = -2Re(lbar * conj(C*Bbar))
  4. CLAMP DI JURY sui coefficienti quantizzati  <-- vedi sotto
  5. export .mem + modello bit-accurate + confronto con il float

------------------------------------------------------------------------------
IL CLAMP DI JURY
------------------------------------------------------------------------------
Quantizzare a1 e a2 in modo indipendente non conserva la posizione del polo. Il
caso critico e' il modo n=0 di S4D-Lin, dove A_imag = 0: il polo e' reale
DOPPIO e la distanza dal cerchio unitario vale

    1 + a1 + a2 = (1 - lbar)^2

cioe' un infinitesimo del SECONDO ordine. Per tau = 1024 vale 9.5e-7: sotto il
quanto di qualunque formato con meno di 21 bit frazionari. L'arrotondamento
porta quindi una radice esattamente su z=1 e il filtro diventa un integratore.

Il rimedio non costa nulla in hardware: si restringono i coefficienti dentro il
triangolo di Jury  (a2 < 1,  |a1| < 1 + a2)  lasciando un quanto di margine.
Verificato su una griglia S4D-Lin realistica: 3/4096 biquad instabili a 24 bit
(75/4096 a 16 bit), zero dopo il clamp, con perturbazione su a1 pari a un
quanto. La costante di tempo resta corretta entro lo 0,4% anche a 16 bit.
"""

import argparse, json, math, os
import numpy as np

# ---------------------------------------------------------------- config
# Architettura ripiegata: NUNITS=8 unita' biquad da NCH=16 canali (era 64x2) e
# NMAC=64 MAC nel mixing (erano TH=256, uno per riga di W_out). Ogni MAC serve
# NPH = TH/NMAC = 4 righe, in NPH fasi da H cicli.
H, MODES, NUNITS, NCH = 128, 32, 8, 16
TH = 2 * H
NMAC = 64
assert TH % NMAC == 0, "TH deve essere multiplo di NMAC"
NPH = TH // NMAC
assert H == NUNITS * NCH, "H deve essere NUNITS*NCH"

W_ACT, F_ACT = 16, 11
W_PAR, F_GAMMA = 16, 13
W_U, F_U = 16, 11
W_A, F_A = 24, 22
W_B, F_B = 16, 15
W_S, F_Y = 32, 19
W_D, F_D = 16, 14
W_BACC = 40
W_SSM, F_SSM = 16, 11
W_W, F_W = 12, 11
W_MACC = 32
F_SIGI, F_SIGO = 11, 14

# SH_B e' PER CANALE, non globale: vedi il commento in s4d_biquad_unit.v
SH_B0 = F_B + F_U - F_Y
SH_D0 = F_D + F_U - F_Y   # nominale; il valore vero e' calibrato in main()
SH_A = F_A
SH_SSM = F_Y - F_SSM
FACC = F_W + F_SSM
RQ_G = FACC - F_SIGI
RQ_A = FACC - F_ACT

EPS_REAL = 1e-5
EPS_NUM = max(1, int(round(EPS_REAL * 2.0 ** (2 * F_ACT) * H ** 2)))


# ---------------------------------------------------------------- fixed point
def shr_round(x, s):
    if s <= 0:
        return x << (-s)
    return (x + (1 << (s - 1))) >> s


def sat(x, w):
    lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    return max(lo, min(hi, x))


def satw(v, s, w=W_S):
    return sat(shr_round(v, s), w)


def calib(x, w, hard=None):
    """Bit frazionari che riempiono w bit senza saturare. Fissarli a occhio e'
    il modo piu' facile di perdere 20 dB senza accorgersene: con F_D = 14 (range
    +-2) i valori di D ~ N(0,1) con coda a 3.5 venivano tosati, e il ramo S4D
    scendeva a 18 dB di SNR."""
    m = float(np.abs(x).max()) if hard is None else hard
    if m <= 0:
        return w - 1
    return int(math.floor(math.log2(((1 << (w - 1)) - 1) / m)))


def q(arr, f, w):
    v = np.round(np.asarray(arr, dtype=np.float64) * 2.0 ** f)
    return np.clip(v, -(1 << (w - 1)), (1 << (w - 1)) - 1).astype(np.int64)


def hx(v, w):
    """Formatta v su w bit in esadecimale, rifiutando i valori che non ci stanno.

    La versione precedente mascherava con `& ((1 << w) - 1)` e restituiva una
    stringa perfettamente valida anche quando il valore era troppo grande: un
    campo in overflow spariva senza lasciare traccia, e l'RTL a valle leggeva
    numeri sbagliati senza alzare nessun flag. E' esattamente cosi' che il campo
    bsh (4 bit, valori calibrati fino a 22) e' rimasto rotto e invisibile.

    Ora l'errore e' rumoroso. Sono accettati sia gli interi senza segno in
    [0, 2^w) sia quelli con segno in [-2^(w-1), 2^(w-1)), che vengono convertiti
    in complemento a due: e' il caso di tutti i tensori quantizzati, che passano
    da q() e sono gia' saturati al formato giusto.
    """
    v = int(v)
    lo, hi = -(1 << (w - 1)), (1 << w) - 1
    if not (lo <= v <= hi):
        raise ValueError(
            f"hx: {v} non entra in {w} bit (ammesso [{lo}, {hi}]). "
            f"Allargare il campo o correggere la calibrazione: NON mascherare."
        )
    return format(v & ((1 << w) - 1), 'x').zfill((w + 3) // 4)


def write(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  {os.path.basename(path):18s} {len(lines):>6d} righe")


# ---------------------------------------------------------------- biquad
def to_biquad(log_A_real, A_imag, log_dt, C):
    lam = -np.exp(log_A_real) + 1j * A_imag
    dt = np.exp(log_dt)[:, None]
    lbar = np.exp(dt * lam)
    Bbar = (lbar - 1) / lam
    r = C * Bbar
    return (-2 * lbar.real, np.abs(lbar) ** 2,
            2 * r.real, -2 * (lbar * np.conj(r)).real, lbar)


def jury_clamp(a1i, a2i, f):
    """Riporta i coefficienti quantizzati dentro il triangolo di stabilita'."""
    one = 1 << f
    a2i = np.minimum(a2i, one - 1)
    lim = one + a2i - 1
    return np.clip(a1i, -lim, lim), a2i


# ---------------------------------------------------------------- PWL
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_pwl(fn, f_out, xh=3, nseg=256, w=16):
    """LUT {slope, base} con interpolazione lineare su [-2^xh, 2^xh)."""
    xs = -(2.0 ** xh) + np.arange(nseg + 1) * (2.0 ** (xh + 1)) / nseg
    ys = q(fn(xs), f_out, w)
    base, slope = ys[:-1], ys[1:] - ys[:-1]
    return base, np.clip(slope, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def pwl_eval(x, base, slope, f_in, f_out, xh=3, nseg=256, log2n=8,
             mode_hi=1, y_lo=0, y_hi=0, w=16):
    fracw = f_in + xh + 1 - log2n
    offs = 1 << (xh + f_in)
    if x < -offs:
        return y_lo
    if x >= offs:
        return sat(x, w) if mode_hi else y_hi
    xs = x + offs
    i, fr = xs >> fracw, xs & ((1 << fracw) - 1)
    return sat(int(base[i]) + shr_round(int(slope[i]) * fr, fracw), w)


# ---------------------------------------------------------------- modello RTL
class LayerModel:
    """Specchio bit-accurate di s4d_layer.v."""

    def __init__(self, p):
        self.p = p
        self.s1 = np.zeros((H, MODES), dtype=object)
        self.s2 = np.zeros((H, MODES), dtype=object)

    def layernorm(self, x):
        p = self.p
        s = int(np.sum(x))
        ssq = int(np.sum(np.asarray(x, dtype=object) ** 2))
        vnum = (ssq << 7) - s * s
        v = vnum + EPS_NUM
        msb = v.bit_length() - 1
        e2 = msb & ~1
        sh = (v >> (e2 - 6)) if e2 >= 6 else (v << (6 - e2))
        inv = p['isqrt_lut'][sh & 0xFF]
        shift = 15 + F_GAMMA - F_U + e2 // 2
        return [sat(shr_round(((int(x[h]) << 7) - s) * inv * int(p['gamma'][h]),
                              shift) + int(p['beta'][h]), W_U)
                for h in range(H)]

    def biquads(self, u):
        p = self.p
        out = []
        for h in range(H):
            uh = int(u[h])
            acc = satw(int(p['D'][h]) * uh, p['sh_d'])      # skip D*u
            for n in range(MODES):
                a1, a2 = int(p['a1'][h, n]), int(p['a2'][h, n])
                b0, b1 = int(p['b0'][h, n]), int(p['b1'][h, n])
                s1, s2 = self.s1[h, n], self.s2[h, n]
                shb = int(p['bsh'][h])
                y = satw(b0 * uh + (s1 << shb), shb)
                a1y, a2y = satw(a1 * y, SH_A), satw(a2 * y, SH_A)
                self.s1[h, n] = satw(satw(b1 * uh, shb) - a1y + s2, 0)
                self.s2[h, n] = -a2y
                acc += y
            out.append(sat(shr_round(acc, SH_SSM), W_SSM))
        return out

    def step(self, x):
        p = self.p
        u = self.layernorm(x)
        yssm = self.biquads(u)
        g = [pwl_eval(v, p['gelu_b'], p['gelu_s'], F_SSM, F_SSM, mode_hi=1)
             for v in yssm]

        acc = [int(p['bout'][c]) for c in range(TH)]
        for c in range(TH):
            for h in range(H):
                acc[c] += int(p['wout'][c, h]) * g[h]
            acc[c] = sat(acc[c], W_MACC)

        out = []
        for h in range(H):
            a = sat(shr_round(acc[h], RQ_A), 16)
            gg = sat(shr_round(acc[H + h], RQ_G), 16)
            sg = pwl_eval(gg, p['sig_b'], p['sig_s'], F_SIGI, F_SIGO,
                          mode_hi=0, y_lo=0, y_hi=1 << F_SIGO)
            out.append(sat(shr_round(a * sg, F_SIGO), W_ACT))
        return [sat(int(x[h]) + out[h], W_ACT) for h in range(H)]


# ---------------------------------------------------------------- float
class LayerFloat:
    """Stesso layer in floating point: riferimento per misurare quanto costa
    davvero il fixed point. Usa gli stessi coefficienti biquad NON quantizzati."""

    def __init__(self, gamma, beta, a1, a2, b0, b1, D, wout, bout):
        self.g, self.b = gamma, beta
        self.a1, self.a2, self.b0, self.b1 = a1, a2, b0, b1
        self.D, self.w, self.bo = D, wout, bout
        self.s1 = np.zeros((H, MODES))
        self.s2 = np.zeros((H, MODES))

    def step(self, x):
        mu, var = x.mean(), x.var()
        u = self.g * (x - mu) / np.sqrt(var + EPS_REAL) + self.b

        uu = u[:, None]
        y = self.b0 * uu + self.s1
        self.s1 = self.b1 * uu - self.a1 * y + self.s2
        self.s2 = -self.a2 * y
        yssm = y.sum(axis=1) + self.D * u

        gl = gelu(yssm)
        v = self.w @ gl + self.bo
        return x + v[:H] * sigmoid(v[H:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='../mem')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- parametri S4D-Lin ------------------------------------------------
    log_A_real = np.log(0.5) + rng.normal(0, 0.3, (H, MODES))
    A_imag = np.pi * np.arange(MODES)[None, :] * np.ones((H, 1))
    log_dt = rng.uniform(np.log(1e-3), np.log(1e-1), H)
    C = (rng.normal(0, 1, (H, MODES)) + 1j * rng.normal(0, 1, (H, MODES))) / np.sqrt(2)
    D = rng.normal(0, 1, H)
    gamma_pre = rng.normal(1.0, 0.15, H)
    gamma = gamma_pre

    a1f, a2f, b0f, b1f, lbar = to_biquad(log_A_real, A_imag, log_dt, C)

    print("range coefficienti:")
    for nm, v in [('a1', a1f), ('a2', a2f), ('b0', b0f), ('b1', b1f)]:
        print(f"  {nm}: [{v.min():+.5f}, {v.max():+.5f}]  max|.| {np.abs(v).max():.5f}")

    # controllo di saturazione su tutti i tensori prima di quantizzare
    print("\ncalibrazione degli esponenti (max|.| -> bit frazionari):")
    for nm, v, w, fx in [('a1', a1f, W_A, F_A), ('a2', a2f, W_A, F_A),
                         ('gamma', gamma_pre, W_PAR, F_GAMMA)]:
        c = calib(v, w)
        flag = '' if c >= fx else f'  <-- F={fx} SATURA, servirebbe {c}'
        print(f"  {nm:7s} max {np.abs(v).max():8.4f}  F consigliato {c:3d}{flag}")

    a1i, a2i = q(a1f, F_A, W_A), q(a2f, F_A, W_A)
    bad = int(((np.abs(a1i) >= (1 << F_A) + a2i) | (a2i >= (1 << F_A))).sum())
    a1i, a2i = jury_clamp(a1i, a2i, F_A)
    good = int(((np.abs(a1i) >= (1 << F_A) + a2i) | (a2i >= (1 << F_A))).sum())
    print(f"\nclamp di Jury: {bad}/{a1i.size} biquad instabili prima, {good} dopo")

    # esponente per canale su b0,b1: riempie i 16 bit sul massimo di quel canale
    bmax = np.maximum(np.abs(b0f), np.abs(b1f)).max(axis=1)
    fb_ch = np.floor(np.log2(((1 << (W_B - 1)) - 1) / np.maximum(bmax, 1e-30))).astype(int)
    fb_ch = np.clip(fb_ch, F_B, F_B + 15)
    bsh = fb_ch + F_U - F_Y                       # shift applicato in hardware
    print(f"esponente b per canale: F_B in [{fb_ch.min()}, {fb_ch.max()}], "
          f"shift in [{bsh.min()}, {bsh.max()}]  (globale sarebbe {SH_B0})")
    print(f"  dinamica di max|b| fra canali: {bmax.max()/bmax.min():.0f}x")

    # Il campo bsh e' largo W_BSH bit (5 dopo l'allargamento; era 4, e i valori
    # sopra 15 venivano troncati in silenzio).
    #
    # Il limite vero non e' il campo ma il datapath: in s4d_biquad_unit la somma
    # e' (b0*u + s1<<bsh) >> bsh su PW = W_A+W_S bit, e con |s1| < 2^(W_S-1)
    # serve W_S + bsh <= PW - 1, cioe' bsh <= W_A - 1 = 23. Il clip di fb_ch a
    # F_B+15 tiene bsh <= 22, un bit sotto il limite.
    W_BSH   = 5
    BSH_MAX = W_A - 1
    assert bsh.max() < (1 << W_BSH), (
        f"bsh max {bsh.max()} non entra in {W_BSH} bit: allargare il campo in "
        f"gen_s4d_mem.py, rtl/s4d_biquad_{{unit,bank}}.v, rtl/s4d_layer.v e nei "
        f"corrispondenti VHDL")
    assert bsh.max() <= BSH_MAX, (
        f"bsh max {bsh.max()} supera il limite del datapath ({BSH_MAX}): "
        f"s1<<bsh esce dai {W_A+W_S} bit di PW e va in overflow silenzioso")
    assert bsh.min() >= SH_B0, (
        f"bsh min {bsh.min()} sotto SH_B0 = {SH_B0}: impossibile per "
        f"costruzione, indica un troncamento a monte")

    b0i = np.stack([q(b0f[h], fb_ch[h], W_B) for h in range(H)])
    b1i = np.stack([q(b1f[h], fb_ch[h], W_B) for h in range(H)])
    f_d = calib(D, W_D)
    print(f"  D       max {np.abs(D).max():8.4f}  F usato       {f_d:3d}"
          f"  (era fisso a {F_D})")
    Di = q(D, f_d, W_D)

    beta = rng.normal(0.0, 0.10, H)
    wout = rng.normal(0, 0.08, (TH, H))
    bout = rng.normal(0, 0.02, TH)
    gi, bi = q(gamma, F_GAMMA, W_PAR), q(beta, F_ACT, W_PAR)
    wi = q(wout, F_W, W_W)
    boi = q(bout, FACC, W_MACC)

    # ---- LUT ---------------------------------------------------------------
    isq = [min(65535, int(round(32768 / math.sqrt((max(i, 64) + .5) / 64))))
           for i in range(256)]
    gb, gs = build_pwl(gelu, F_SSM)
    sb, ss = build_pwl(sigmoid, F_SIGO)

    print(f"\nscrittura in {os.path.abspath(a.out)}")
    write(os.path.join(a.out, 'isqrt_lut.mem'), [hx(v, 16) for v in isq])
    write(os.path.join(a.out, 'ln_gamma.mem'), [hx(v, 16) for v in gi])
    write(os.path.join(a.out, 'ln_beta.mem'), [hx(v, 16) for v in bi])
    write(os.path.join(a.out, 'gelu_lut.mem'),
          [hx((int(s) & 0xFFFF) << 16 | (int(b) & 0xFFFF), 32) for b, s in zip(gb, gs)])
    write(os.path.join(a.out, 'sigmoid_lut.mem'),
          [hx((int(s) & 0xFFFF) << 16 | (int(b) & 0xFFFF), 32) for b, s in zip(sb, ss)])
    # D e shift di b viaggiano insieme: {bsh[W_BSH-1:0], D[W_D-1:0]}
    # Il campo bsh e' passato da 4 a 5 bit: la posizione non cambia (parte
    # sempre dal bit W_D), quindi i file scritti con 4 bit restano leggibili.
    write(os.path.join(a.out, 's4d_D.mem'),
          [hx((int(bsh[h]) << W_D) | (int(Di[h]) & ((1 << W_D) - 1)), W_D + W_BSH)
           for h in range(H)])

    # coefficienti: ordine (unita', slot, modo); l'unita' j serve i canali
    # j*NCH .. j*NCH+NCH-1. Sviluppando idx = j*NCH*MODES + slot*MODES + n con
    # h = j*NCH + slot si ottiene idx = h*MODES + n: il file NON dipende da
    # NUNITS, ed e' invariante rispetto alla ripiegatura del banco.
    lines = []
    for j in range(NUNITS):
        for slot in range(NCH):
            h = j * NCH + slot
            for n in range(MODES):
                word = ((int(b1i[h, n]) & 0xFFFF) << 64 |
                        (int(b0i[h, n]) & 0xFFFF) << 48 |
                        (int(a2i[h, n]) & 0xFFFFFF) << 24 |
                        (int(a1i[h, n]) & 0xFFFFFF))
                lines.append(hx(word, 80))
    write(os.path.join(a.out, 's4d_coef.mem'), lines)

    # W_out, formato del mixing ripiegato: NPH*H righe da NMAC*W_W bit,
    #     riga(r*H + h) [m*W_W +: W_W] = wout[r*NMAC + m][h]
    # con h in ordine NATURALE. Il banco ripiegato riordina le proprie uscite in
    # un buffer ed emette 0..127 in sequenza, quindi qui non c'e' nessuna
    # permutazione: prima le righe erano ordinate pari-poi-dispari perche' era
    # quello l'ordine di emissione del banco 64x2.
    # Le H righe da TH*W_W = 3072 bit di prima non stavano in nessuna porta di
    # BRAM e finivano in registri distribuiti.
    rows = []
    for r in range(NPH):
        for h in range(H):
            word = 0
            for m in range(NMAC):
                word |= (int(wi[r * NMAC + m, h]) & ((1 << W_W) - 1)) << (m * W_W)
            rows.append(hx(word, NMAC * W_W))
    write(os.path.join(a.out, 's4d_wout.mem'), rows)
    write(os.path.join(a.out, 's4d_bout.mem'), [hx(v, W_MACC) for v in boi])

    # ---- modello bit-accurate ---------------------------------------------
    p = dict(isqrt_lut=isq, gamma=gi, beta=bi, a1=a1i, a2=a2i, b0=b0i, b1=b1i,
             D=Di, sh_d=f_d + F_U - F_Y, bsh=bsh, wout=wi, bout=boi, gelu_b=gb, gelu_s=gs, sig_b=sb, sig_s=ss)
    m = LayerModel(p)

    x = q(rng.normal(0, 1.5, (a.steps, H)), F_ACT, W_ACT)
    outs, stim = [], []
    for k in range(a.steps):
        stim.extend(int(v) for v in x[k])
        outs.append(m.step(x[k]))

    write(os.path.join(a.out, 's4d_stim.mem'), [hx(v, W_ACT) for v in stim])
    write(os.path.join(a.out, 's4d_exp.mem'),
          [hx(v, W_ACT) for row in outs for v in row])

    o = np.array(outs, dtype=np.float64) / 2 ** F_ACT
    print(f"\nuscita del layer (fixed point): min {o.min():+.4f} max {o.max():+.4f} "
          f"rms {np.sqrt((o**2).mean()):.4f}")
    st = np.array([[int(v) for v in x[k]] for k in range(a.steps)]) / 2 ** F_ACT
    print(f"ingresso:                       min {st.min():+.4f} max {st.max():+.4f} "
          f"rms {np.sqrt((st**2).mean()):.4f}")

    sat_out = int((np.abs(np.array(outs)) >= (1 << (W_ACT - 1)) - 1).sum())
    print(f"campioni in saturazione all'uscita: {sat_out}/{a.steps*H}")

    # ---- confronto con il riferimento float --------------------------------
    # DUE riferimenti, per separare due errori che si sommano ma hanno cause
    # diverse:
    #   (A) coefficienti esatti -> include lo spostamento dei poli dovuto alla
    #       quantizzazione di a1,a2. Per filtri risonanti ad alto Q un polo
    #       spostato di un quanto sfasa progressivamente l'uscita: e' un errore
    #       inevitabile, non un difetto del datapath.
    #   (B) stessi coefficienti quantizzati -> isola l'aritmetica del datapath
    #       (stato, arrotondamenti, saturazioni, LUT). E' questo il numero che
    #       dice se l'RTL e' dimensionato bene.
    fl = LayerFloat(gamma, beta, a1f, a2f, b0f, b1f, D, wout, bout)
    ref = np.array([fl.step(x[k] / 2.0 ** F_ACT) for k in range(a.steps)])

    flq = LayerFloat(gi / 2.0**F_GAMMA, bi / 2.0**F_ACT,
                     a1i / 2.0**F_A, a2i / 2.0**F_A,
                     b0i / 2.0**fb_ch[:, None], b1i / 2.0**fb_ch[:, None],
                     Di / 2.0**f_d, wi / 2.0**F_W, boi / 2.0**FACC)
    refq = np.array([flq.step(x[k] / 2.0 ** F_ACT) for k in range(a.steps)])
    err = o - ref
    print("\n--- fixed point vs float ---")
    print(f"errore assoluto max  = {np.abs(err).max():.5f}")
    print(f"errore rms           = {np.sqrt((err**2).mean()):.5f}")
    print(f"rms del segnale      = {np.sqrt((ref**2).mean()):.5f}")
    print(f"SNR                  = {20*np.log10(np.sqrt((ref**2).mean())/max(np.sqrt((err**2).mean()),1e-12)):.1f} dB")

    # contributo del solo ramo S4D (l'uscita e' dominata dal residuo)
    br_f = ref - x[:a.steps] / 2.0 ** F_ACT
    br_q = o   - x[:a.steps] / 2.0 ** F_ACT
    be = br_q - br_f
    xr = x[:a.steps] / 2.0 ** F_ACT
    br_fq = refq - xr

    def snr(sig, err):
        return 20*np.log10(np.sqrt((sig**2).mean()) /
                           max(np.sqrt((err**2).mean()), 1e-12))

    print(f"\nramo S4D (l'uscita totale e' dominata dal residuo):")
    print(f"  rms del ramo                     {np.sqrt((br_f**2).mean()):.5f}")
    print(f"  (A) vs coefficienti esatti       SNR {snr(br_f, br_q-br_f):5.1f} dB"
          f"   <- include la quantizzazione dei coefficienti")
    print(f"  (B) vs stessi coeff quantizzati  SNR {snr(br_fq, br_q-br_fq):5.1f} dB"
          f"   <- solo aritmetica del datapath")
    print(f"  quota dovuta ai coefficienti     SNR {snr(br_f, br_fq-br_f):5.1f} dB")

    with open(os.path.join(a.out, 's4d_config.vh'), 'w') as f:
        f.write("// generato da gen_s4d_mem.py -- non modificare a mano\n")
        for k, v in [('TB_H', H), ('TB_MODES', MODES), ('TB_NUNITS', NUNITS),
                     ('TB_NCH', NCH), ('TB_NMAC', NMAC), ('TB_NPH', NPH),
                     ('TB_STEPS', a.steps), ('TB_EPS_NUM', EPS_NUM),
                     ('TB_F_A', F_A), ('TB_F_B', F_B), ('TB_F_Y', F_Y)]:
            f.write(f"localparam integer {k} = {v};\n")

    json.dump({'jury_before': bad, 'jury_after': good, 'EPS_NUM': EPS_NUM,
               'sat_out': sat_out},
              open(os.path.join(a.out, 's4d_report.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
