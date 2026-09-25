#!/usr/bin/env python3
"""
s4d_export.py -- motore condiviso fra i notebook: quantizzazione, packing dei
.mem, modello bit-accurate, calibrazione dei range.

Perche' esiste. La prima versione del notebook di addestramento aveva tutto
inline. Appena e' servito un secondo notebook (la riparazione dei range) le
copie sarebbero diventate due, e sarebbero divergute alla prima modifica --
esattamente il problema che il README della repo descrive per ln_gamma.mem,
scritto da due generatori diversi con contenuti diversi. Qui c'e' una copia
sola, importata da entrambi.

I formati NON sono ridefiniti qui: arrivano da gen_s4d_mem.py, che e' lo
specchio verificato del Verilog. Questo modulo li usa, non li ricopia.

Dipendenze: numpy e gen_s4d_mem (obbligatori), torch (solo per il modello e la
calibrazione dei range; l'export funziona senza).
"""

import math
import os
import sys

import numpy as np

import gen_s4d_mem as G

# Versione dell'interfaccia. Serve perche' i moduli vivono su Drive e il
# notebook li carica solo se MANCANO: una copia vecchia di questo file accanto a
# un s4d_bn.py nuovo non da' un errore all'import, lo da' venti minuti dopo, a
# meta' della cella di riscalatura, con un TypeError su un argomento keyword.
#   1: versione LayerNorm originale
#   2: parametri opzionali per la norm (with_norm, norm, norms/norm_final),
#      necessari a s4d_bn.py
API = 2

H, MODES, NUNITS, NCH, TH = G.H, G.MODES, G.NUNITS, G.NCH, G.TH
# NMAC / NPH: ripiegatura del mixing. NMAC MAC in parallelo, ognuno responsabile
# di NPH = TH/NMAC righe di W_out, servite in NPH fasi da H cicli.
NMAC, NPH = G.NMAC, G.NPH

# campo dell'esponente per canale di b, impacchettato con D
W_BSH = 5
BSH_MAX = G.W_A - 1          # limite del datapath: s1<<bsh deve stare in PW bit

# encoder
W_WENC, W_BENC, W_OUT_ENC, F_OUT_ENC = 18, 32, 16, 11

# decoder
W_CIN, W_CLS, W_ACC_D, K_CLS = 8, 8, 32, 10

LOG2_H = 7

# tetti dei formati, in unita' reali
CEIL_ACT = 2.0 ** (G.W_ACT - 1 - G.F_ACT)     # residuo e ramo "a": Q5.11 -> 16
CEIL_SSM = 2.0 ** (G.W_SSM - 1 - G.F_SSM)     # uscita banco biquad


# ============================================================================
# Formati
# ============================================================================
def set_f_ssm(new_f_ssm):
    """Cambia F_SSM e ricalcola i derivati dentro gen_s4d_mem.

    F_SSM e' un semplice generic di s4d_layer (default 11) e i costanti che ne
    dipendono sono derivati DENTRO l'RTL:
        s4d_biquad_bank : SH_O = F_Y - F_OUT
        s4d_mixing_glu  : FACC = F_W + F_IN, RQ_G = FACC - F_SIGI,
                          RQ_A = FACC - F_A, SH_O = F_A + F_SIGO - F_OUT
    Quindi cambiarlo NON e' una modifica al codice VHDL: e' un generic da
    passare. Ma i .mem generati dipendono dal valore (bout e' quantizzato a
    FACC bit, la LUT GELU e' costruita su F_SSM), quindi se l'RTL viene
    elaborato con il default e i .mem sono stati generati con un altro valore
    il risultato e' sbagliato in silenzio. Vedi required_generics().
    """
    G.F_SSM = int(new_f_ssm)
    G.SH_SSM = G.F_Y - G.F_SSM
    G.FACC = G.F_W + G.F_SSM
    G.RQ_G = G.FACC - G.F_SIGI
    G.RQ_A = G.FACC - G.F_ACT
    globals()["CEIL_SSM"] = 2.0 ** (G.W_SSM - 1 - G.F_SSM)
    return G.F_SSM


def formats_table():
    return "\n".join([
        f"attivazioni / residuo  Q{16-G.F_ACT}.{G.F_ACT}  W_ACT={G.W_ACT}   tetto +-{CEIL_ACT:g}",
        f"uscita banco (yssm)    Q{16-G.F_SSM}.{G.F_SSM}  W_SSM={G.W_SSM}   tetto +-{CEIL_SSM:g}",
        f"a1,a2                  W_A={G.W_A} F_A={G.F_A}     tetto +-{2**(G.W_A-1-G.F_A)}",
        f"b0,b1                  W_B={G.W_B} F_B={G.F_B}+    esponente PER CANALE",
        f"W_out                  W_W={G.W_W} F_W={G.F_W}     tetto +-{2**(G.W_W-1-G.F_W)}",
        f"gamma                  W_PAR={G.W_PAR} F_GAMMA={G.F_GAMMA}",
        f"accumulatore mixing    W_MACC={G.W_MACC} FACC={G.FACC}  RQ_A={G.RQ_A} RQ_G={G.RQ_G}",
    ])


# ============================================================================
# Quantizzazione di un layer
# ============================================================================
def quantize_layer(d, verbose=True, f_d=None):
    """d: dict con log_A_real, A_imag, log_dt, C, D, wout, bout, gamma, beta.

    f_d: forza i bit frazionari di D invece di calibrarli su questo layer.
    Serve perche' nell'RTL `F_D` e' un generic GLOBALE -- `SH_D = F_D + F_U -
    F_Y` e' una costante di s4d_biquad_unit, una sola per tutto il design --
    mentre `G.calib` lo sceglie per layer e non da' lo stesso valore. Esportare
    D con un f_d per layer e poi passare all'RTL un F_D solo significa scalare
    il ramo D di una potenza di due su tutti i layer che non hanno quel valore.
    Misurato: SNR sui logit da 33 dB a 5-7 dB, e la classe predetta cambia.
    Il chiamante deve passare min(f_d) su tutti i layer: il minimo perche' un
    f_d piu' alto del calibrato tosa D, che e' molto peggio di un bit in meno.

    gamma/beta sono OPZIONALI: se mancano (variante BatchNorm, dove la norm e'
    un affine per canale quantizzato da s4d_bn.quantize_affine) il resto del
    layer viene quantizzato lo stesso. Tutto cio' che sta a valle della norm --
    biquad, D, mixing, GLU -- non dipende da quale normalizzazione si usa, ed e'
    esattamente il motivo per cui questa funzione non e' duplicata nel modulo BN.
    """
    a1f, a2f, b0f, b1f, lbar = G.to_biquad(d["log_A_real"], d["A_imag"],
                                           d["log_dt"], d["C"])
    rho = float(np.abs(lbar).max())

    # --- clamp di Jury ------------------------------------------------------
    # a1 e a2 quantizzati indipendentemente escono dal triangolo di stabilita'.
    # Il caso critico e' il modo n=0 (polo reale doppio): 1+a1+a2 = (1-lbar)^2,
    # un infinitesimo del secondo ordine che l'arrotondamento annulla, e il
    # biquad diventa un integratore.
    a1i, a2i = G.q(a1f, G.F_A, G.W_A), G.q(a2f, G.F_A, G.W_A)
    bad = int(((np.abs(a1i) >= (1 << G.F_A) + a2i) | (a2i >= (1 << G.F_A))).sum())
    a1i, a2i = G.jury_clamp(a1i, a2i, G.F_A)
    good = int(((np.abs(a1i) >= (1 << G.F_A) + a2i) | (a2i >= (1 << G.F_A))).sum())

    # --- esponente di b per canale -----------------------------------------
    bmax = np.maximum(np.abs(b0f), np.abs(b1f)).max(axis=1)
    fb_ch = np.floor(np.log2(((1 << (G.W_B - 1)) - 1) / np.maximum(bmax, 1e-30))).astype(int)
    fb_ch = np.clip(fb_ch, G.F_B, G.F_B + 15)
    bsh = fb_ch + G.F_U - G.F_Y
    assert bsh.max() < (1 << W_BSH), f"bsh {bsh.max()} non entra in {W_BSH} bit"
    assert bsh.max() <= BSH_MAX, f"bsh {bsh.max()} > {BSH_MAX}: s1<<bsh esce dal datapath"
    assert bsh.min() >= G.SH_B0, f"bsh min {bsh.min()} < SH_B0 {G.SH_B0}"

    b0i = np.stack([G.q(b0f[h], fb_ch[h], G.W_B) for h in range(H)])
    b1i = np.stack([G.q(b1f[h], fb_ch[h], G.W_B) for h in range(H)])

    # --- tensori restanti ---------------------------------------------------
    f_d_cal = G.calib(d["D"], G.W_D)
    f_d = f_d_cal if f_d is None else int(f_d)
    qz = dict(
        a1=a1i, a2=a2i, b0=b0i, b1=b1i, bsh=bsh, fb_ch=fb_ch,
        D=G.q(d["D"], f_d, G.W_D), f_d=f_d, f_d_cal=f_d_cal,
        sh_d=f_d + G.F_U - G.F_Y,
        wout=G.q(d["wout"], G.F_W, G.W_W),
        bout=G.q(d["bout"], G.FACC, G.W_MACC),
        jury_before=bad, jury_after=good, rho=rho,
    )
    if "gamma" in d:
        qz["gamma"] = G.q(d["gamma"], G.F_GAMMA, G.W_PAR)
        qz["beta"] = G.q(d["beta"], G.F_ACT, G.W_PAR)

    # --- saturazione: rumorosa, mai mascherata ------------------------------
    warn = []
    checks = [("D", d["D"], G.W_D, f_d),
              ("wout", d["wout"], G.W_W, G.F_W),
              ("bout", d["bout"], G.W_MACC, G.FACC)]
    if "gamma" in d:
        checks = [("gamma", d["gamma"], G.W_PAR, G.F_GAMMA),
                  ("beta", d["beta"], G.W_PAR, G.F_ACT)] + checks
    for nm, arr, w, f in checks:
        lim = (1 << (w - 1)) - 1
        over = np.abs(np.asarray(arr))[np.abs(np.round(np.asarray(arr) * 2.0 ** f)) >= lim]
        if over.size:
            warn.append(f"{nm}: {over.size}/{np.asarray(arr).size} pesi tosati dal "
                        f"formato Q{w-f}.{f} (tetto {lim/2**f:.4f}, "
                        f"max|.|={np.abs(arr).max():.4f})")
    qz["warn"] = warn

    if verbose:
        print(f"  max |lambda_bar| {rho:.6f}   Jury {bad} -> {good} fuori dal triangolo")
        print(f"  F_B per canale [{fb_ch.min()}, {fb_ch.max()}], bsh [{bsh.min()}, {bsh.max()}]"
              f"   dinamica di max|b| fra canali {bmax.max()/bmax.min():.0f}x")
        note = "" if f_d == f_d_cal else f"  (calibrato {f_d_cal}, forzato al globale)"
        print(f"  F_D {f_d}  (max|D| {np.abs(d['D']).max():.4f}, "
              f"riempie {np.abs(d['D']).max()*2**f_d:.0f}/{(1<<(G.W_D-1))-1}){note}")
        for w_ in warn:
            print("  ATTENZIONE  " + w_)
    return qz


# ============================================================================
# Scrittura dei .mem
# ============================================================================
def _write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return f"  {os.path.basename(str(path)):24s} {len(lines):>6d} righe"


def write_layer_mem(m, qz, outdir, verbose=True, with_norm=True):
    """with_norm=False salta ln_gamma/ln_beta: la variante BatchNorm scrive al
    loro posto l<m>_bn_w / l<m>_bn_b (vedi s4d_bn.write_affine_mem), che NON
    sono la stessa cosa e non devono finire nella stessa cartella."""
    pre, out = f"l{m}_", str(outdir)
    j = lambda n: os.path.join(out, pre + n)
    msgs = []
    if with_norm:
        msgs += [
            _write(j("ln_gamma.mem"), [G.hx(v, G.W_PAR) for v in qz["gamma"]]),
            _write(j("ln_beta.mem"), [G.hx(v, G.W_PAR) for v in qz["beta"]]),
        ]
    msgs += [
        # {bsh[W_BSH-1:0], D[W_D-1:0]}: D e il suo shift viaggiano insieme
        _write(j("s4d_D.mem"),
               [G.hx((int(qz["bsh"][h]) << G.W_D) | (int(qz["D"][h]) & ((1 << G.W_D) - 1)),
                     G.W_D + W_BSH) for h in range(H)]),
    ]
    # coefficienti: ordine (unita' j, slot, modo n), lo stesso con cui
    # s4d_biquad_unit indirizza la propria memoria (cnt = slot*MODES + n)
    lines = []
    for u in range(NUNITS):
        for slot in range(NCH):
            h = u * NCH + slot
            for n in range(MODES):
                lines.append(G.hx(
                    (int(qz["b1"][h, n]) & 0xFFFF) << 64 |
                    (int(qz["b0"][h, n]) & 0xFFFF) << 48 |
                    (int(qz["a2"][h, n]) & 0xFFFFFF) << 24 |
                    (int(qz["a1"][h, n]) & 0xFFFFFF), 80))
    msgs.append(_write(j("s4d_coef.mem"), lines))

    # W_out: NPH*H righe da NMAC*W_W bit, la riga (r*H + h) e' la fase r del
    # canale di ingresso h. Vedi write_wout_mem().
    msgs.append(write_wout_mem(j("s4d_wout.mem"), qz["wout"]))
    msgs.append(_write(j("s4d_bout.mem"), [G.hx(v, G.W_MACC) for v in qz["bout"]]))
    if verbose:
        print("\n".join(msgs))


def wout_row_perm():
    """Permutazione delle COLONNE di W_out, oggi l'identita'.

    Restava qui la convenzione di ordinamento fra export e RTL, e resta qui
    anche adesso che non c'e' piu' niente da permutare: e' il punto in cui la
    convenzione e' documentata, non un residuo da cancellare.

    Com'era. Il banco biquad spaziale (NUNITS=64 unita' da NCH=2 canali)
    emetteva i canali nell'ordine in cui erano pronti -- l'unita' j serviva i
    canali 2j e 2j+1, quindi prima tutti i pari 0,2,...,126 e poi tutti i
    dispari 1,3,...,127. L'export scriveva le righe di s4d_wout.mem in
    quell'ordine, cosi' che il mixing potesse consumarle di seguito senza
    indirizzamento.

    Com'e' adesso. Il banco ripiegato (NUNITS=8 unita' da NCH=16) riordina i
    risultati in un buffer ed emette i canali in ordine naturale 0..127. Il
    riordino avviene in HARDWARE: qui non serve piu', e la permutazione e'
    l'identita'.

    Se il buffer venisse rimosso dall'RTL questa funzione deve tornare a
    contenere la permutazione corrispondente al NUNITS di allora, cioe'
    l'ordine in cui il banco emette i canali:

        [h for slot in range(NCH) for h in range(slot, H, NCH)]

    che per NCH=2 e' esattamente pari-poi-dispari. Attenzione: con il formato
    attuale la permutazione andrebbe applicata all'indice h DENTRO la riga
    (r*H + h), non all'ordine delle righe come faceva la versione vecchia.
    """
    return list(range(H))


def write_wout_mem(path, wout):
    """Scrive s4d_wout.mem nel formato del mixing ripiegato e lo rilegge.

    Formato. NPH*H = 512 righe da NMAC*W_W = 768 bit. La riga (r*H + h)
    contiene i pesi degli NMAC canali di uscita serviti dalla fase r, tutti
    relativi allo stesso canale di ingresso h:

        riga(r*H + h) [m*W_W +: W_W] = W_out[r*NMAC + m][h]

    con r = 0..NPH-1, h = 0..H-1 in ordine naturale, m = 0..NMAC-1.

    Perche' non le H righe da TH*W_W = 3072 bit di prima: 3072 bit non stanno
    in nessuna porta di BRAM, e il sintetizzatore realizzava la memoria in
    registri distribuiti -- decine di migliaia di LUT per un dato che sta in
    poche BRAM. 768 bit sono NMAC letture parallele da una porta ragionevole,
    ed e' esattamente quanto il mixing consuma in un ciclo: NMAC MAC, ciascuno
    responsabile di NPH righe di W_out, in NPH fasi da H cicli.

    La rilettura non e' un lusso. Una permutazione sbagliata qui non produce
    ne' overflow ne' instabilita': produce una rete che calcola una funzione
    diversa con statistiche del tutto normali, e non la si intercetta se non
    confrontando i logit -- cioe' molto piu' tardi, se mai.
    """
    wout = np.asarray(wout, dtype=np.int64)
    if wout.shape != (TH, H):
        raise ValueError(f"wout ha forma {wout.shape}, attesa {(TH, H)}")
    perm = wout_row_perm()
    rows = []
    for r in range(NPH):
        for h in perm:
            word = 0
            for m in range(NMAC):
                word |= (int(wout[r * NMAC + m, h]) & ((1 << G.W_W) - 1)) << (m * G.W_W)
            rows.append(G.hx(word, NMAC * G.W_W))
    msg = _write(path, rows)
    check_wout_mem(path, wout)
    return msg


def read_wout_mem(path):
    """Ricostruisce W_out (TH, H) da s4d_wout.mem con gli indici di bit dell'RTL."""
    words = [int(t, 16) for t in open(str(path)).read().split()]
    if len(words) != NPH * H:
        raise ValueError(f"{path}: {len(words)} righe, attese {NPH*H}")
    mask, half = (1 << G.W_W) - 1, 1 << (G.W_W - 1)
    inv = np.empty(H, dtype=np.int64)          # posizione -> canale
    inv[:] = wout_row_perm()
    out = np.zeros((TH, H), dtype=np.int64)
    for idx, word in enumerate(words):
        if word >> (NMAC * G.W_W):
            raise ValueError(f"{path}: riga {idx} eccede {NMAC*G.W_W} bit")
        r, pos = divmod(idx, H)
        h = int(inv[pos])
        for m in range(NMAC):
            v = (word >> (m * G.W_W)) & mask
            out[r * NMAC + m, h] = v - (1 << G.W_W) if v >= half else v
    return out


def check_wout_mem(path, wout):
    """Rilegge s4d_wout.mem e lo confronta con la matrice quantizzata di partenza."""
    got = read_wout_mem(path)
    want = np.asarray(wout, dtype=np.int64)
    if got.shape != want.shape or not np.array_equal(got, want):
        bad = np.argwhere(got != want)
        c, h = (int(bad[0][0]), int(bad[0][1])) if bad.size else (-1, -1)
        raise ValueError(
            f"{os.path.basename(str(path))}: la rilettura non corrisponde a W_out. "
            f"{len(bad)}/{want.size} pesi diversi, il primo in (c={c}, h={h}): "
            f"scritto {got[c, h]} invece di {want[c, h]}. "
            f"Packing o permutazione sbagliati: NON e' un errore di quantizzazione, "
            f"e a valle non si vede se non nei logit.")
    return True


def quantize_encoder(enc_w, enc_b, mu, sigma, verbose=True):
    """Ripiega la normalizzazione dentro i pesi: l'FPGA riceve il pixel grezzo.

        u = (p/255 - mu)/sigma,  x = w u + b
        =>  x = (w/(255 sigma)) p + (b - w mu/sigma)

    Il prodotto w'*p e' esatto (p e' intero): l'unico arrotondamento e' finale.
    """
    w_fold = np.asarray(enc_w) / (255.0 * sigma)
    b_fold = np.asarray(enc_w) * (-mu / sigma) + np.asarray(enc_b)
    f_enc = min(G.calib(w_fold, W_WENC), G.calib(b_fold, W_BENC))
    sh_out = f_enc - F_OUT_ENC
    if sh_out < 0:
        raise ValueError("SH_OUT negativo: i pesi non riempiono il formato del residuo")
    wq, bq = G.q(w_fold, f_enc, W_WENC), G.q(b_fold, f_enc, W_BENC)
    if verbose:
        print(f"  max|w'| {np.abs(w_fold).max():.3e}  max|b'| {np.abs(b_fold).max():.4f}")
        print(f"  F_WENC {f_enc}   SH_OUT {sh_out}")
    return dict(w=wq, b=bq, f_enc=f_enc, sh_out=sh_out)


def write_encoder_mem(enc, outdir, verbose=True):
    msgs = [_write(os.path.join(str(outdir), "enc_w.mem"), [G.hx(v, W_WENC) for v in enc["w"]]),
            _write(os.path.join(str(outdir), "enc_b.mem"), [G.hx(v, W_BENC) for v in enc["b"]])]
    if verbose:
        print("\n".join(msgs))


def write_decoder_mem(nf_gamma, nf_beta, cls_w, cls_b, f_cin, outdir,
                      nlayer=6, verbose=True, with_norm=True):
    """LN finale (indice nlayer per mem_name) + classificatore INT8.

    with_norm=False scrive solo il classificatore: nella variante BatchNorm la
    norm finale e' un affine e la scrive s4d_bn.write_decoder_affine_mem.
    In quel caso nf_gamma/nf_beta possono essere None.
    """
    f_w = int(math.floor(math.log2(((1 << (W_CLS - 1)) - 1) / float(np.abs(cls_w).max()))))
    rq_shift = G.F_ACT - f_cin
    if rq_shift < 0:
        raise ValueError("RQ_SHIFT negativo: xbar non entra nel formato INT8")
    gq = G.q(nf_gamma, G.F_GAMMA, G.W_PAR) if with_norm else None
    bq = G.q(nf_beta, G.F_ACT, G.W_PAR) if with_norm else None
    wq = G.q(cls_w, f_w, W_CLS)
    bcq = G.q(cls_b, f_w + f_cin, W_ACC_D)
    o = str(outdir)
    rows = []
    for h in range(H):
        word = 0
        for c in range(K_CLS):
            word |= (int(wq[c, h]) & 0xFF) << (c * W_CLS)
        rows.append(G.hx(word, K_CLS * W_CLS))
    msgs = []
    if with_norm:
        msgs += [_write(os.path.join(o, f"l{nlayer}_ln_gamma.mem"), [G.hx(v, G.W_PAR) for v in gq]),
                 _write(os.path.join(o, f"l{nlayer}_ln_beta.mem"), [G.hx(v, G.W_PAR) for v in bq])]
    msgs += [_write(os.path.join(o, "cls_w.mem"), rows),
             _write(os.path.join(o, "cls_b.mem"), [G.hx(v, W_ACC_D) for v in bcq])]
    if verbose:
        print("\n".join(msgs))
    return dict(gamma=gq, beta=bq, w=wq, b=bcq, f_w=f_w, f_cin=f_cin, rq_shift=rq_shift)


def write_luts(outdir, verbose=True):
    """isqrt, gelu, sigmoid: non dipendono dai pesi, ma la GELU dipende da F_SSM."""
    isq = [min(65535, int(round(32768 / math.sqrt((max(i, 64) + .5) / 64)))) for i in range(256)]
    gb, gs = G.build_pwl(G.gelu, G.F_SSM)
    sb, ss = G.build_pwl(G.sigmoid, G.F_SIGO)
    o = str(outdir)
    pack = lambda b, s: [G.hx((int(y) & 0xFFFF) << 16 | (int(x) & 0xFFFF), 32) for x, y in zip(b, s)]
    msgs = [_write(os.path.join(o, "isqrt_lut.mem"), [G.hx(v, 16) for v in isq]),
            _write(os.path.join(o, "gelu_lut.mem"), pack(gb, gs)),
            _write(os.path.join(o, "sigmoid_lut.mem"), pack(sb, ss))]
    if verbose:
        print("\n".join(msgs))
    return dict(isqrt=np.asarray(isq, np.int64),
                gelu_b=np.asarray(gb, np.int64), gelu_s=np.asarray(gs, np.int64),
                sig_b=np.asarray(sb, np.int64), sig_s=np.asarray(ss, np.int64))


# ============================================================================
# Modello bit-accurate, vettorizzato (specchio di s4d_layer)
# ============================================================================
def shr_r(x, s):
    """Shift aritmetico con arrotondamento. s scalare o array, anche negativo."""
    s = np.asarray(s)
    if s.ndim == 0:
        s = int(s)
        return x << (-s) if s <= 0 else (x + (1 << (s - 1))) >> s
    sp = np.maximum(s, 1)
    pos = (x + (np.int64(1) << (sp - 1))) >> sp
    neg = x << np.maximum(-s, 0)
    return np.where(s > 0, pos, neg)


def satn(x, w):
    return np.clip(x, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def satw(v, s, w=None):
    return satn(shr_r(v, s), G.W_S if w is None else w)


def isqrt_n(v, lut):
    """Replica di s4d_isqrt. v > 0 e < 2^53, cosi' frexp da' l'MSB esatto."""
    _, e = np.frexp(v.astype(np.float64))
    msb = e.astype(np.int64) - 1
    e2 = msb & ~np.int64(1)
    sh = np.where(e2 >= 6, np.right_shift(v, np.maximum(e2 - 6, 0)),
                  np.left_shift(v, np.maximum(6 - e2, 0)))
    return lut[sh & 0xFF], e2 // 2


def pwl_n(x, base, slope, f_in, xh=3, log2n=8, mode_hi=True, y_lo=0, y_hi=0, w=16):
    fracw = f_in + xh + 1 - log2n
    offs = 1 << (xh + f_in)
    xs = np.clip(x, -offs, offs - 1) + offs
    i, fr = xs >> fracw, xs & ((1 << fracw) - 1)
    y = satn(base[i] + shr_r(slope[i] * fr, fracw), w)
    y = np.where(x < -offs, y_lo, y)
    return np.where(x >= offs, satn(x, w) if mode_hi else y_hi, y)


def layernorm_n(x, gamma, beta, f_out, lut, eps_num=None):
    eps_num = G.EPS_NUM if eps_num is None else eps_num
    s = x.sum(1, keepdims=True)
    ssq = (x * x).sum(1, keepdims=True)
    vnum = (ssq << LOG2_H) - s * s
    inv, half_e = isqrt_n(vnum + eps_num, lut)
    sh = (15 + G.F_GAMMA - f_out) + half_e
    p = ((x << LOG2_H) - s) * inv * gamma[None, :]
    return satn(shr_r(p, sh) + beta[None, :], G.W_ACT)


class LayerFX:
    """Specchio bit-accurate di s4d_layer, con il batch come prima dimensione.

    Conta anche DOVE satura: il residuo e il ramo "a" sono i due siti in cui un
    modello addestrato esce da Q5.11, e senza contarli separatamente non si
    capisce quale correzione serve.
    """

    def __init__(self, qz, luts, B, norm=None):
        """norm: callable (B,H) int64 -> (B,H) int64 in formato F_U che sostituisce
        il LayerNorm. Serve alla variante BatchNorm, dove la norm e' un affine per
        canale: tutto il resto del datapath e' identico e non va duplicato."""
        self.q, self.lut = qz, luts
        self.norm = norm
        self.s1 = np.zeros((B, H, MODES), dtype=np.int64)
        self.s2 = np.zeros((B, H, MODES), dtype=np.int64)
        self.sat_x = self.sat_a = self.sat_y = 0
        self.acc_bound = 0

    def step(self, x):
        q, lut = self.q, self.lut
        u = (self.norm(x) if self.norm is not None
             else layernorm_n(x, q["gamma"], q["beta"], G.F_U, lut["isqrt"]))

        uh = u[:, :, None]
        shb = q["bsh"][None, :, None]
        y = satw(q["b0"][None] * uh + np.left_shift(self.s1, shb), shb)
        a1y = satw(q["a1"][None] * y, G.SH_A)
        a2y = satw(q["a2"][None] * y, G.SH_A)
        self.s1 = satw(satw(q["b1"][None] * uh, shb) - a1y + self.s2, 0)
        self.s2 = -a2y
        acc = satw(q["D"][None, :] * u, q["sh_d"]) + y.sum(2)
        raw = shr_r(acc, G.SH_SSM)
        self.sat_y += int((np.abs(raw) > (1 << (G.W_SSM - 1)) - 1).sum())
        yssm = satn(raw, G.W_SSM)

        g = pwl_n(yssm, lut["gelu_b"], lut["gelu_s"], G.F_SSM, mode_hi=True)

        z = g @ q["wout"].T + q["bout"][None, :]
        # se il bound sulle somme parziali sta sotto 2^31 nessuna saturazione
        # intermedia dell'accumulatore a W_MACC bit e' possibile
        self.acc_bound = max(self.acc_bound, int(
            (np.abs(g) @ np.abs(q["wout"]).T + np.abs(q["bout"])[None, :]).max()))
        z = satn(z, G.W_MACC)

        raw_a = shr_r(z[:, :H], G.RQ_A)
        self.sat_a += int((np.abs(raw_a) > (1 << 15) - 1).sum())
        a = satn(raw_a, 16)
        gg = satn(shr_r(z[:, H:], G.RQ_G), 16)
        sg = pwl_n(gg, lut["sig_b"], lut["sig_s"], G.F_SIGI,
                   mode_hi=False, y_lo=0, y_hi=1 << G.F_SIGO)
        out = satn(shr_r(a * sg, G.F_SIGO), G.W_ACT)
        xn = x + out
        self.sat_x += int((np.abs(xn) > (1 << (G.W_ACT - 1)) - 1).sum())
        return satn(xn, G.W_ACT)


def encoder_n(pix, enc):
    acc = enc["w"][None, :] * pix[:, None] + enc["b"][None, :]
    return satn(shr_r(acc, enc["sh_out"]), W_OUT_ENC)


def run_chain_fx(pixels, QZ, enc, dec_gamma, dec_beta, luts, log2_l=None,
                 norms=None, norm_final=None):
    """pixels: (B, L) interi 0..255. Ritorna (xbar, layers).

    norms / norm_final: liste di callable che sostituiscono il LayerNorm dei sei
    layer e quello del decoder. Con i default (None) il comportamento e' quello
    di prima, bit per bit: i notebook LayerNorm non cambiano.
    """
    B, L = pixels.shape
    log2_l = int(math.log2(L)) if log2_l is None else log2_l
    layers = [LayerFX(q, luts, B, norm=(norms[i] if norms is not None else None))
              for i, q in enumerate(QZ)]
    nf = (norm_final if norm_final is not None else
          (lambda v: layernorm_n(v, dec_gamma, dec_beta, G.F_ACT, luts["isqrt"])))
    pool = np.zeros((B, H), dtype=np.int64)
    for k in range(L):
        x = encoder_n(pixels[:, k], enc)
        for lay in layers:
            x = lay.step(x)
        pool += nf(x)
    return satn(shr_r(pool, log2_l), G.W_ACT), layers


def classify_fx(xbar, dec):
    xq = satn(shr_r(xbar, dec["rq_shift"]), W_CIN)
    return satn(xq @ dec["w"].T + dec["b"][None, :], W_ACC_D)


# ============================================================================
# Riscalatura del ramo del valore
# ============================================================================
def rescale_value_branch(W, s):
    """Scala di s il residuo SENZA cambiare la funzione calcolata.

    LayerNorm e' invariante di scala, quindi scalare x non cambia il suo
    ingresso normalizzato. Serve percio' scalare anche l'uscita di ogni layer,
    e questo si fa scalando SOLO il ramo "a" di output_linear (righe 0:H) e
    lasciando intatto quello del gate: la sigmoide non e' omogenea, scalare il
    gate cambierebbe la funzione. Con

        enc_w, enc_b        *= s
        wout[0:H], bout[0:H] *= s     per ogni layer

    si ottiene x_m -> s*x_m per ogni m, e siccome anche norm_f e' invariante di
    scala i logit restano identici. L'unica differenza e' l'eps=1e-5 di
    LayerNorm, che non e' invariante: con residui di ampiezza >> 1e-2 l'effetto
    e' sotto il millesimo (misurato: 2.6e-3 su logit di ampiezza 26 con s=1/2).

    s dovrebbe essere una potenza di due: cosi' la riscalatura e' un semplice
    spostamento dell'esponente e non aggiunge errore di arrotondamento ai pesi.
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in W.items() if k != "layers"}
    out["enc_w"] = W["enc_w"] * s
    out["enc_b"] = W["enc_b"] * s
    out["layers"] = []
    for d in W["layers"]:
        e = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        e["wout"] = d["wout"].copy()
        e["bout"] = d["bout"].copy()
        e["wout"][:H] *= s
        e["bout"][:H] *= s
        out["layers"].append(e)
    return out


def choose_scale(max_act, ceiling=None, fill=0.75):
    """Potenza di due piu' grande che porta max_act sotto fill*tetto.

    Potenza di due e non un fattore qualunque: cosi' la riscalatura sposta solo
    l'esponente e non aggiunge errore ai pesi. Il prezzo e' che si arrotonda per
    difetto, quindi il margine finale sta fra fill*tetto/2 e fill*tetto.
    """
    ceiling = CEIL_ACT if ceiling is None else ceiling
    if max_act <= 0:
        return 1.0
    return 2.0 ** min(0, int(math.floor(math.log2(fill * ceiling / max_act))))


def needed_f_ssm(max_yssm, fill=0.75):
    """F_SSM piu' grande (quindi piu' preciso) che tiene max_yssm sotto il tetto."""
    lim = (1 << (G.W_SSM - 1)) - 1
    # mai sopra 11: e' il default dell'RTL, alzarlo non servirebbe a niente e
    # renderebbe i .mem incompatibili con un'elaborazione di default
    return min(11, int(math.floor(math.log2(fill * lim / max(max_yssm, 1e-12)))))


# ============================================================================
# Modello PyTorch (identico a CIFAR/Train_CIFAR_s4.py) e calibrazione
# ============================================================================
def torch_model(d_model=128, n_layers=6, d_state=64, dropout=0.2, n_classes=10):
    """Costruisce il classificatore. Importa torch solo se chiamata."""
    import torch
    import torch.nn as nn

    class S4DKernel(nn.Module):
        def __init__(self, dm, N=64, dt_min=1e-3, dt_max=1e-1):
            super().__init__()
            Hh, n = dm, N // 2
            log_dt = torch.rand(Hh) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            C = torch.randn(Hh, n, dtype=torch.cfloat)
            self.C = nn.Parameter(torch.view_as_real(C))
            self.log_dt = nn.Parameter(log_dt)
            self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(Hh, n)))
            self.A_imag = nn.Parameter(math.pi * torch.arange(n).repeat(Hh, 1).float())

        def forward(self, L):
            dt = torch.exp(self.log_dt)
            C = torch.view_as_complex(self.C)
            A = -torch.exp(self.log_A_real) + 1j * self.A_imag
            dtA = A * dt.unsqueeze(-1)
            K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
            Ce = C * (torch.exp(dtA) - 1.0) / A
            return 2 * torch.einsum('hn,hnl->hl', Ce, torch.exp(K)).real

    class S4D(nn.Module):
        def __init__(self, dm, d_state=64, dropout=0.0):
            super().__init__()
            self.h, self.n = dm, d_state
            self.D = nn.Parameter(torch.randn(self.h))
            self.kernel = S4DKernel(self.h, N=self.n)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2 * self.h, kernel_size=1), nn.GLU(dim=-2))

        def forward(self, u):
            L = u.size(-1)
            k = self.kernel(L)
            y = torch.fft.irfft(torch.fft.rfft(u, n=2 * L) * torch.fft.rfft(k, n=2 * L),
                                n=2 * L)[..., :L]
            y = y + u * self.D.unsqueeze(-1)
            y = self.dropout(self.activation(y))
            return self.output_linear(y)

    class S4DsCIFARClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.layers = nn.ModuleList([S4D(d_model, d_state) for _ in range(n_layers)])
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.norm_f = nn.LayerNorm(d_model)
            self.head_dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model, n_classes)

        def forward(self, x):
            x = self.input_proj(x)
            for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
                res = x
                x = norm(x).transpose(1, 2)
                x = layer(x).transpose(1, 2)
                x = res + drop(x)
            return self.classifier(self.head_dropout(self.norm_f(x).mean(dim=1)))

    return S4DsCIFARClassifier()


def weights_to_numpy(model, n_layers=6):
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    W = dict(enc_w=sd["input_proj.weight"].reshape(-1), enc_b=sd["input_proj.bias"].reshape(-1),
             nf_gamma=sd["norm_f.weight"], nf_beta=sd["norm_f.bias"],
             cls_w=sd["classifier.weight"], cls_b=sd["classifier.bias"], layers=[])
    for m in range(n_layers):
        p = f"layers.{m}."
        c = sd[p + "kernel.C"]
        W["layers"].append(dict(
            log_A_real=sd[p + "kernel.log_A_real"], A_imag=sd[p + "kernel.A_imag"],
            log_dt=sd[p + "kernel.log_dt"], C=c[..., 0] + 1j * c[..., 1],
            D=sd[p + "D"], wout=sd[p + "output_linear.0.weight"][:, :, 0],
            bout=sd[p + "output_linear.0.bias"],
            gamma=sd[f"norms.{m}.weight"], beta=sd[f"norms.{m}.bias"]))
    return W


def numpy_to_torch_(model, W, with_norm=True):
    """Riscrive i pesi del modello torch a partire dal dizionario numpy.

    with_norm=False salta gamma/beta: nella variante BatchNorm le norm hanno
    quattro tensori invece di due e le riscrive s4d_bn.numpy_to_torch_. Tutto il
    resto -- kernel, D, mixing, encoder, classificatore -- e' identico e passa
    di qui.
    """
    import torch
    sd = model.state_dict()
    sd["input_proj.weight"].copy_(torch.tensor(W["enc_w"].reshape(-1, 1)))
    sd["input_proj.bias"].copy_(torch.tensor(W["enc_b"]))
    if with_norm:
        sd["norm_f.weight"].copy_(torch.tensor(W["nf_gamma"]))
        sd["norm_f.bias"].copy_(torch.tensor(W["nf_beta"]))
    sd["classifier.weight"].copy_(torch.tensor(W["cls_w"]))
    sd["classifier.bias"].copy_(torch.tensor(W["cls_b"]))
    for m, d in enumerate(W["layers"]):
        p = f"layers.{m}."
        sd[p + "kernel.log_A_real"].copy_(torch.tensor(d["log_A_real"]))
        sd[p + "kernel.A_imag"].copy_(torch.tensor(d["A_imag"]))
        sd[p + "kernel.log_dt"].copy_(torch.tensor(d["log_dt"]))
        sd[p + "kernel.C"].copy_(torch.tensor(np.stack([d["C"].real, d["C"].imag], -1)))
        sd[p + "D"].copy_(torch.tensor(d["D"]))
        sd[p + "output_linear.0.weight"].copy_(torch.tensor(d["wout"][:, :, None]))
        sd[p + "output_linear.0.bias"].copy_(torch.tensor(d["bout"]))
        if with_norm:
            sd[f"norms.{m}.weight"].copy_(torch.tensor(d["gamma"]))
            sd[f"norms.{m}.bias"].copy_(torch.tensor(d["beta"]))
    return model


def calibrate_ranges(model, loader, device, n_images=128, n_layers=6, quantile=None):
    """Misura i range delle attivazioni sul modello VERO, con degli hook.

    Reimplementare la catena in numpy per misurare i range avrebbe voluto dire
    misurare una reimplementazione: gli hook osservano i moduli che verranno
    esportati, quindi non c'e' modello intermedio di cui fidarsi.

    Siti osservati, che sono esattamente i tre che possono uscire dal formato:
      x     ingresso di norms[m] e di norm_f -> il residuo, Q5.11
      yssm  ingresso della GELU              -> uscita del banco biquad, Q5.11
      a     prima meta' di output_linear[0]  -> ramo del valore prima del GLU
    """
    import torch
    Hh = model.norm_f.normalized_shape[0]
    acc = dict(x=[0.0] * (n_layers + 1), yssm=[0.0] * n_layers, a=[0.0] * n_layers,
               over_x=[0] * (n_layers + 1), over_y=[0] * n_layers, over_a=[0] * n_layers,
               n_x=0, n_y=0, n_a=0)
    hs = []

    def pre_x(i):
        def f(_, inp):
            v = inp[0].detach()
            acc["x"][i] = max(acc["x"][i], v.abs().max().item())
            acc["over_x"][i] += int((v.abs() > CEIL_ACT).sum().item())
        return f

    def pre_y(i):
        def f(_, inp):
            v = inp[0].detach()
            acc["yssm"][i] = max(acc["yssm"][i], v.abs().max().item())
            acc["over_y"][i] += int((v.abs() > CEIL_SSM).sum().item())
        return f

    def post_a(i):
        def f(_, __, out):
            v = out.detach()[:, :Hh]
            acc["a"][i] = max(acc["a"][i], v.abs().max().item())
            acc["over_a"][i] += int((v.abs() > CEIL_ACT).sum().item())
        return f

    for m in range(n_layers):
        hs.append(model.norms[m].register_forward_pre_hook(pre_x(m)))
        hs.append(model.layers[m].activation.register_forward_pre_hook(pre_y(m)))
        hs.append(model.layers[m].output_linear[0].register_forward_hook(post_a(m)))
    hs.append(model.norm_f.register_forward_pre_hook(pre_x(n_layers)))

    model.eval()
    seen = 0
    try:
        with torch.no_grad():
            for x, _ in loader:
                if seen >= n_images:
                    break
                x = x[: n_images - seen].to(device)
                model(x)
                b, L = x.shape[0], x.shape[1]
                acc["n_x"] += b * L * Hh
                acc["n_y"] += b * L * Hh
                acc["n_a"] += b * L * Hh
                seen += b
    finally:
        for h in hs:
            h.remove()
    acc["n_images"] = seen
    acc["max_x"] = max(acc["x"])
    acc["max_a"] = max(acc["a"])
    acc["max_yssm"] = max(acc["yssm"])
    return acc


def print_ranges(r, n_layers=6):
    print(f"range misurati su {r['n_images']} immagini   "
          f"(tetto residuo/a +-{CEIL_ACT:g}, tetto yssm +-{CEIL_SSM:g})")
    print(f"{'':9s}{'max|yssm|':>11s}{'oltre':>9s}{'max|a|':>10s}{'oltre':>9s}"
          f"{'max|x|':>10s}{'oltre':>9s}")
    for m in range(n_layers):
        print(f"layer {m:<3d}{r['yssm'][m]:11.2f}{100*r['over_y'][m]/max(r['n_y'],1):8.2f}%"
              f"{r['a'][m]:10.2f}{100*r['over_a'][m]/max(r['n_a'],1):8.2f}%"
              f"{r['x'][m+1]:10.2f}{100*r['over_x'][m+1]/max(r['n_x'],1):8.2f}%")


def required_generics(enc, dec, f_ssm=None, nlayer=6, mem_dir="mem_trained/"):
    """I generic che DEVONO essere passati a s4d_top: i default non bastano.

    Ognuno di questi ha un default che era giusto per i pesi casuali del banco
    di prova e non lo e' per pesi addestrati. Passarne uno sbagliato non alza
    nessun flag: il datapath lavora in un formato e i .mem sono in un altro.

    F_SSM merita una nota. Fino a poco fa s4d_top NON lo esponeva: cablava
    `W_SSM => W_ACT, F_SSM => F_ACT` verso s4d_layer, cioe' dava per scontato
    che l'uscita del banco biquad stesse nello stesso formato del residuo.
    E' vero solo per coincidenza, e la coincidenza cade appena i pesi
    addestrati fanno uscire yssm dal range. Ora e' un generic con default
    16/11, che riproduce il comportamento di prima.
    """
    f_ssm = G.F_SSM if f_ssm is None else f_ssm
    rows = [
        ("MEM_DIR", f'"{mem_dir}"', None),
        ("PER_LAYER_MEM", "true", "i sei layer hanno pesi diversi"),
        ("WITH_LOADER", "true", "false in sintesi definitiva"),
        ("NUNITS", NUNITS, "banco ripiegato: %d unita' (era 64)" % NUNITS),
        ("NCH", NCH, "%d canali per unita' (era 2)" % NCH),
        ("NMAC", NMAC, "mixing ripiegato: %d MAC, NPH = TH/NMAC = %d fasi"
                       % (NMAC, NPH)),
        ("EPS_NUM", G.EPS_NUM, "default 16384"),
        ("F_WENC", enc["f_enc"], "default 23; SH_OUT = F_WENC-F_ACT = %d" % enc["sh_out"]),
        ("F_SSM", f_ssm, "default 11"),
        ("RQ_SHIFT", dec["rq_shift"], "default 7"),
    ]
    out = ["generic map ("]
    for k, v, c in rows:
        line = f"    {k:<14s}=> {v},"
        out.append(line if c is None else f"{line:<44s}-- {c}")
    out.append(")")
    out += ["", f"-- decoder: F_CIN {dec['f_cin']}, F_W {dec['f_w']}",
            "-- i default sopra sono quelli del banco di prova a pesi casuali:",
            "-- con questi .mem sono tutti sbagliati, e nessuno se ne accorge."]
    return "\n".join(out)
