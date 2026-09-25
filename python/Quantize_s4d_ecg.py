#!/usr/bin/env python3
"""
Quantize_s4d_ecg.py -- riscalatura, quantizzazione, export dei .mem e verifica
bit-accurate del modello ECG (PTB-XL) addestrato da Train_s4d_ecg.py.

Si lancia senza argomenti, dalla radice del repository:

    conda run -n mamba_env python Quantize_s4d_ecg.py

E' la stessa catena di Train_and_quantize_s4d_sCIFAR.py (variante BatchNorm) e
di Quantize_s4d_speech_command.py, e usa gli STESSI moduli gen_s4d_mem /
s4d_export / s4d_bn estratti dal monolitico, senza modificarli:

  1. calibrazione dei range con gli hook sul modello vero (E.calibrate_ranges)
  2. scala globale s, potenza di due, sul ramo del valore, compensata esattamente
     su w della BN ripiegata (B.rescale_value_branch) + controllo in ulp
  3. layer S4D: biquad ZOH, clamp di Jury, esponente di b per canale, D, W_out,
     bout (E.quantize_layer); BN ripiegata in affine (B.quantize_affine)
  4. LUT GELU/sigmoide (E.write_luts)
  5. verifica bit-accurate con E.LayerFX + B.AffineFX, cioe' lo specchio
     verificato contro l'RTL

DIFFERENZE RISPETTO A sCIFAR, imposte dall'hardware ECG e NON dal modello
Il bitstream S4D_PTB_XL_UltraTiny e' gia' sintetizzato. Dal block design
(s4d_PTB_XL.bd) e dai default dell'RTL i generic sono FISSI:

    F_SSM   = 9    (s4d_core_axis_0, impostato nel bd)
    F_D     = 13   (default di s4d_core_axis)
    F_WENC  = 31   (default di s4d_encoder_ecg_axis) -> pesi encoder a 20 bit
                   frazionari, bias a 31, SH_OUT = 20
    RQ_SHIFT= 6    (default di s4d_decoder_axis)     -> F_CIN = 5
    BN_SH   = 13   (s4d_decoder_axis_0, impostato nel bd) -> norm finale F_BNW 13
    K       = 5, L = 1024, LOG2_L = 10

sCIFAR li SCEGLIE dalla calibrazione e poi stampa i generic da passare; qui,
con HW_LOCK = True, si IMPONGONO quelli del bitstream e si misura quanto costano
(saturazioni, SNR, accuratezza). Lo shift della BN dei sei layer non e' un
generic: viaggia nello stream dei pesi (ld_bn_sh), quindi resta calibrato per
layer come in sCIFAR. Con HW_LOCK = False si ottiene la quantizzazione libera
di sCIFAR, come riferimento: quei .mem NON girano sul bitstream attuale.

L'encoder ECG e' diverso da quello di sCIFAR (12 ingressi Q5.11 con segno
invece di un pixel uint8) e ha qui il suo quantizzatore, scritto su
s4d_encoder_ecg_axis.v: y = sat16(round((sum_c W[ch][c]*x[c] + b[ch]) >> SH_OUT)),
enc_w.mem = 128 righe da 12x18 bit con c0 nei bit bassi, enc_b.mem a 32 bit.

NOTA SULLA SCHEDA
enc_w/enc_b, l6_bn_*, cls_w/cls_b e le LUT sono ROM inizializzate da $readmemh:
cambiano solo risintetizzando (stessi generic, nessuna modifica RTL). I pesi
dei sei layer passano invece dallo stream (Export_s4d_wstream.py, con MEM_DIR
puntato alla cartella prodotta qui).
"""

import json
import math
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import Train_s4d_ecg as T

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
# Esperimento da quantizzare: primo argomento da riga di comando, altrimenti
# il run multi-label piu' recente scritto da Train_s4d_ecg.py in experiments/.
#   python Quantize_s4d_ecg.py experiments/exp_YYYYMMDD_HHMM_ecg100_l1024_fixed_multi
def _pick_experiment():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    runs = sorted(Path("experiments").glob("exp_*_ecg100_l1024_*_multi"))
    if not runs:
        sys.exit("nessun esperimento trovato: lanciare prima Train_s4d_ecg.py, "
                 "oppure passare la cartella dell'esperimento come argomento")
    return runs[-1]


EXP = _pick_experiment()
CKPT = EXP / "best.pt"

HW_LOCK = True          # True: generic del bitstream ECG; False: liberi (sCIFAR)
HW = dict(F_SSM=9, F_D=13, F_WENC=31, RQ_SHIFT=6, BN_SH_FINAL=13, K=5, L=1024)

FILL = 0.75             # frazione del tetto dopo la riscalatura, come sCIFAR
N_CAL = None            # record di validazione per la calibrazione (None = tutti)
N_EVAL_FX = None        # record di test per la verifica bit-accurate (None = tutti)
CHUNK_FX = 32           # record per processo nella verifica bit-accurate
WORKERS_FX = 32         # processi paralleli (la catena bit-accurate e' su CPU)
BATCH = 32
SEED = 1337
WRITE_MEM = True        # False: solo misura, i .mem vanno in una cartella temporanea


# ============================================================================
# modello <-> numpy (encoder a 12 ingressi)
# ============================================================================
def weights_to_numpy_ecg(B, model):
    """B.weights_to_numpy appiattisce enc_w (fatto per un ingresso solo):
    qui si rimette la forma (H, 12)."""
    W = B.weights_to_numpy(model, T.N_LAYERS)
    sd = model.state_dict()
    W["enc_w"] = sd["input_proj.weight"].detach().cpu().double().numpy()
    W["enc_b"] = sd["input_proj.bias"].detach().cpu().double().numpy()
    return W


def numpy_to_torch_ecg(B, model, W):
    """B.numpy_to_torch_ scrive enc_w come colonna (H, 1): lo si sovrascrive."""
    W1 = dict(W)
    W1["enc_w"] = np.zeros(W["enc_w"].shape[0])
    B.numpy_to_torch_(model, W1)
    with torch.no_grad():
        model.input_proj.weight.copy_(torch.tensor(W["enc_w"]))
        model.input_proj.bias.copy_(torch.tensor(W["enc_b"]))
    return model


# ============================================================================
# encoder ECG: quantizzazione, .mem, specchio bit-accurate
# ============================================================================
W_WENC, W_BENC, W_ENCACC, NIN = 18, 32, 40, 12


def quantize_encoder_ecg(G, enc_w, enc_b, f_wenc=None):
    """acc = sum_c wq[ch][c]*xq[c] + bq[ch], con xq in Q5.11: F_WENC = f_w + 11.

    f_wenc None: il piu' grande che non satura ne' w (18 bit) ne' b (32 bit),
    come quantize_encoder di sCIFAR. Altrimenti imposto (generic del bitstream).
    """
    w = np.asarray(enc_w, dtype=np.float64)
    b = np.asarray(enc_b, dtype=np.float64)
    f_free = min(G.calib(w, W_WENC) + G.F_ACT, G.calib(b, W_BENC))
    f = f_free if f_wenc is None else int(f_wenc)
    f_w = f - G.F_ACT
    sh_out = f - G.F_ACT
    lim_w, lim_b = (1 << (W_WENC - 1)) - 1, (1 << (W_BENC - 1)) - 1
    sat_w = int((np.abs(np.round(w * 2.0 ** f_w)) > lim_w).sum())
    sat_b = int((np.abs(np.round(b * 2.0 ** f)) > lim_b).sum())
    return dict(w=G.q(w, f_w, W_WENC), b=G.q(b, f, W_BENC), f_enc=f, f_w=f_w,
                sh_out=sh_out, f_free=f_free, sat_w=sat_w, sat_b=sat_b,
                max_w=float(np.abs(w).max()), max_b=float(np.abs(b).max()))


def write_encoder_mem_ecg(G, enc, outdir):
    rows = []
    for h in range(enc["w"].shape[0]):
        word = 0
        for c in range(NIN):                          # C0_AT_LSB = 1
            word |= (int(enc["w"][h, c]) & ((1 << W_WENC) - 1)) << (c * W_WENC)
        rows.append(G.hx(word, NIN * W_WENC))
    (outdir / "enc_w.mem").write_text("\n".join(rows) + "\n")
    (outdir / "enc_b.mem").write_text("\n".join(G.hx(v, W_BENC) for v in enc["b"]) + "\n")


def encoder_ecg_n(E, G, xq, enc):
    """xq (B, 12) int64 Q5.11 -> (B, H) Q5.11, come s4d_encoder_ecg_axis.v (S2..S6)."""
    acc = xq @ enc["w"].T + enc["b"][None, :]
    lim = 1 << (W_ENCACC - 1)
    assert np.abs(acc).max() < lim, "accumulatore dell'encoder oltre W_ENCACC"
    return E.satn(E.shr_r(acc, enc["sh_out"]), G.W_ACT)


# ============================================================================
# quantizzazione della norm finale con SH imposto
# ============================================================================
def quantize_affine_fixed_sh(G, B, w, b, sh, f_out, name):
    """B.quantize_affine con F_BNW = sh + f_out - F_ACT imposto (BN_SH del bd)."""
    f_bnw = sh + f_out - G.F_ACT
    f_b = f_bnw + G.F_ACT
    w = np.asarray(w, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    lim_w, lim_b = (1 << (B.W_BNW - 1)) - 1, (1 << (B.W_BNB - 1)) - 1
    warn = []
    nw = int((np.abs(np.round(w * 2.0 ** f_bnw)) > lim_w).sum())
    nb = int((np.abs(np.round(b * 2.0 ** f_b)) > lim_b).sum())
    if nw:
        warn.append(f"{name}: {nw}/{w.size} w tosati con F_BNW {f_bnw} "
                    f"(max|w| {np.abs(w).max():.3f}, tetto {lim_w / 2 ** f_bnw:.3f})")
    if nb:
        warn.append(f"{name}: {nb}/{b.size} b tosati con F_B {f_b}")
    wq, bq = G.q(w, f_bnw, B.W_BNW), G.q(b, f_b, B.W_BNB)
    nz = np.abs(w) > 0
    rel = (0.5 / 2.0 ** f_bnw) / np.abs(w[nz]).min() if nz.any() else 0.0
    acc = int(np.abs(wq).max()) * ((1 << (G.W_ACT - 1)) - 1) + int(np.abs(bq).max())
    qa = dict(w=wq, b=bq, f_bnw=f_bnw, f_b=f_b, f_out=f_out, sh=sh, warn=warn,
              max_w=float(np.abs(w).max()), max_b=float(np.abs(b).max()),
              acc_bound=acc, rel_worst=float(rel),
              f_bnw_free=B.quantize_affine(w, b, f_out=f_out, name=name,
                                           verbose=False)["f_bnw"])
    print(f"  {name}: F_BNW {f_bnw} (imposto; libero {qa['f_bnw_free']})  SH {sh}  "
          f"max|w| {qa['max_w']:.4f}  max|b| {qa['max_b']:.4f}")
    for x in warn:
        print("  ATTENZIONE  " + x)
    return qa


# ============================================================================
# verifica bit-accurate, parallela
# ============================================================================
_CTX = {}


def _chain_worker(args):
    """Catena intera su un blocco di record: encoder -> 6 layer -> norm finale
    -> mean pool. Gira in un processo figlio (fork): legge i pesi da _CTX."""
    i0, i1 = args
    E, G, B = _CTX["E"], _CTX["G"], _CTX["B"]
    X = np.asarray(_CTX["X"][i0:i1], dtype=np.int64)          # (b, L, 12)
    b, L, _ = X.shape
    norms = [B.AffineFX(qa) for qa in _CTX["QA"]]            # W_OUT = W_U
    nf = B.AffineFX(_CTX["NF"], w_out=G.W_ACT)
    layers = [E.LayerFX(q, _CTX["luts"], b, norm=n) for q, n in zip(_CTX["QZ"], norms)]
    pool = np.zeros((b, G.H), dtype=np.int64)
    enc_sat = 0
    lim = (1 << (G.W_ACT - 1)) - 1
    for k in range(L):
        acc = X[:, k] @ _CTX["enc"]["w"].T + _CTX["enc"]["b"][None, :]
        raw = E.shr_r(acc, _CTX["enc"]["sh_out"])
        enc_sat += int((np.abs(raw) > lim).sum())
        x = E.satn(raw, G.W_ACT)
        for lay in layers:
            x = lay.step(x)
        pool += nf(x)
    xbar = E.satn(E.shr_r(pool, int(math.log2(L))), G.W_ACT)
    sat = dict(enc=enc_sat,
               layers=[dict(sat_x=l.sat_x, sat_a=l.sat_a, sat_y=l.sat_y,
                            acc_bound=l.acc_bound, norm=n.sat)
                       for l, n in zip(layers, norms)],
               final=nf.sat)
    return i0, xbar, sat


def run_chain_parallel(idx_blocks):
    with mp.get_context("fork").Pool(WORKERS_FX) as pool:
        out, done = [], 0
        for r in pool.imap_unordered(_chain_worker, idx_blocks):
            out.append(r)
            done += r[1].shape[0]
            print(f"  bit-accurate {done}/{idx_blocks[-1][1]}", end="\r", flush=True)
    print()
    out.sort(key=lambda r: r[0])
    xbar = np.concatenate([r[1] for r in out])
    sat = dict(enc=sum(r[2]["enc"] for r in out), final=sum(r[2]["final"] for r in out),
               layers=[])
    for m in range(T.N_LAYERS):
        ls = [r[2]["layers"][m] for r in out]
        sat["layers"].append(dict(
            sat_x=sum(l["sat_x"] for l in ls), sat_a=sum(l["sat_a"] for l in ls),
            sat_y=sum(l["sat_y"] for l in ls), norm=sum(l["norm"] for l in ls),
            acc_bound=max(l["acc_bound"] for l in ls)))
    return xbar, sat


# ============================================================================
# metriche
# ============================================================================
def metrics(logits, ym, y):
    aucs = [T.auc(ym[:, c], logits[:, c]) for c in range(len(T.CLASSES))]
    one = y >= 0
    p = logits[one].argmax(1)
    k = len(T.CLASSES)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y[one], p), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return dict(macro_auc=float(np.nanmean(aucs)),
                auc={c: a for c, a in zip(T.CLASSES, aucs)},
                acc=float(np.diag(cm).sum() / max(cm.sum(), 1)),
                bal_acc=float(rec.mean()), cm=cm.tolist(),
                n_records=int(len(y)), n_single=int(one.sum()))


def auc_table(title, ym, m):
    n = len(ym)
    lines = [f"=== {title} ===",
             f"{'classe':6s} {'positivi':>9s} {'negativi':>9s} {'AUC':>8s}"]
    for k, c in enumerate(T.CLASSES):
        pz = int(ym[:, k].sum())
        lines.append(f"{c:6s} {pz:9d} {n - pz:9d} {m['auc'][c]:8.4f}")
    lines.append(f"{'macro':6s} {'':9s} {'':9s} {m['macro_auc']:8.4f}")
    return "\n".join(lines)


# ============================================================================
# main
# ============================================================================
def main():
    G, E, B = T.load_modules()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    assert T.FS == 100 and T.L == HW["L"], "Train_s4d_ecg.py deve essere a FS=100, L=1024"
    E.K_CLS = HW["K"]

    rep_train = json.loads((EXP / "report.json").read_text())
    cache = T.CACHE_DIR.with_name(f"{T.CACHE_DIR.name}_{rep_train['norm_mode']}")
    if rep_train.get("label_mode") == "multi":
        cache = cache.with_name(cache.name + "_multi")
    print(f"checkpoint {CKPT}\ncache {cache}   HW_LOCK {HW_LOCK}")

    stamp = time.strftime("%Y%m%d_%H%M")
    tag = "hw" if HW_LOCK else "free"
    out_dir = EXP / f"quant_{stamp}_{tag}"
    mem_dir = out_dir / "mem" if WRITE_MEM else Path("/tmp") / f"ecg_quant_dry_{stamp}"
    mem_dir.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = T.ecg_model(B, len(T.CLASSES)).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
    model.eval()

    # --- calibrazione sul validation (fold 9) ------------------------------
    class CalDS(torch.utils.data.Dataset):          # calibrate_ranges vuole (x, y)
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            x, _, y = self.ds[i]
            return x, y
    val = T.ECGDataset(cache / "val_x.npy", cache / "val_y.npy", cache / "val_ym.npy")
    n_cal = len(val) if N_CAL is None else min(N_CAL, len(val))
    cal = torch.utils.data.DataLoader(CalDS(val), batch_size=BATCH, shuffle=False)
    W = weights_to_numpy_ecg(B, model)
    rng_before = E.calibrate_ranges(model, cal, dev, n_images=n_cal, n_layers=T.N_LAYERS)
    print(f"\ncalibrazione su {n_cal} record di validazione. Range prima della riscalatura:")
    E.print_ranges(rng_before, T.N_LAYERS)

    s = E.choose_scale(max(rng_before["max_x"], rng_before["max_a"]), fill=FILL)
    W_fix = B.rescale_value_branch(W, s)

    # --- la riscalatura non deve cambiare la funzione -----------------------
    model_fix = numpy_to_torch_ecg(B, T.ecg_model(B, len(T.CLASSES)), W_fix).to(dev).eval()
    xb, _ = next(iter(cal))
    with torch.no_grad():
        l0 = model(xb.to(dev)).double().cpu().numpy()
        l1 = model_fix(xb.to(dev)).double().cpu().numpy()
    d = float(np.abs(l1 - l0).max())
    ulp = float(np.spacing(np.float32(max(np.abs(l0).max(), 1.0))))
    print(f"\ns = {s}   controllo riscalatura: scarto max {d:.3e} ({d/ulp:.0f} ulp)")
    assert d < 256 * ulp, "la riscalatura ha cambiato la funzione"

    rng_after = E.calibrate_ranges(model_fix, cal, dev, n_images=n_cal, n_layers=T.N_LAYERS)
    print("\nrange dopo la riscalatura:")
    E.print_ranges(rng_after, T.N_LAYERS)

    # --- F_SSM e F_D --------------------------------------------------------
    f_ssm_free = E.needed_f_ssm(rng_before["max_yssm"], fill=FILL)
    f_ssm = HW["F_SSM"] if HW_LOCK else f_ssm_free
    E.set_f_ssm(f_ssm)
    f_d_cal = [G.calib(dd["D"], G.W_D) for dd in W_fix["layers"]]
    F_D = HW["F_D"] if HW_LOCK else min(f_d_cal)
    print(f"\nF_SSM {f_ssm} (calibrato {f_ssm_free}; tetto yssm +-{E.CEIL_SSM:g}, "
          f"max|yssm| misurato {rng_before['max_yssm']:.2f})")
    print(f"F_D   {F_D} (calibrati per layer {f_d_cal})")

    luts = E.write_luts(mem_dir, verbose=False)
    (mem_dir / "isqrt_lut.mem").unlink()            # senza LayerNorm non serve

    # --- layer -------------------------------------------------------------
    QZ, QA = [], []
    for m, dd in enumerate(W_fix["layers"]):
        print(f"\nlayer {m}")
        q = E.quantize_layer(dd, verbose=True, f_d=F_D)
        E.write_layer_mem(m, q, mem_dir, verbose=False, with_norm=False)
        qa = B.quantize_affine(dd["bn_w"], dd["bn_b"], name=f"l{m}_bn", verbose=True)
        B.write_affine_mem(m, qa, mem_dir, verbose=False)
        QZ.append(q); QA.append(qa)

    # --- encoder -----------------------------------------------------------
    enc = quantize_encoder_ecg(G, W_fix["enc_w"], W_fix["enc_b"],
                               HW["F_WENC"] if HW_LOCK else None)
    print(f"\nencoder: F_WENC {enc['f_enc']} (libero {enc['f_free']})  SH_OUT {enc['sh_out']}  "
          f"max|w| {enc['max_w']:.4f}  max|b| {enc['max_b']:.4f}  "
          f"tosati w {enc['sat_w']}/{enc['w'].size}  b {enc['sat_b']}/{enc['b'].size}")
    write_encoder_mem_ecg(G, enc, mem_dir)

    # --- norm finale --------------------------------------------------------
    print()
    if HW_LOCK:
        nf_qa = quantize_affine_fixed_sh(G, B, W_fix["nf_w"], W_fix["nf_b"],
                                         HW["BN_SH_FINAL"], G.F_ACT, f"l{T.N_LAYERS}_bn")
        B.write_affine_mem(T.N_LAYERS, nf_qa, mem_dir, verbose=False)
    else:
        nf_qa = B.write_decoder_affine_mem(W_fix["nf_w"], W_fix["nf_b"], mem_dir,
                                           nlayer=T.N_LAYERS, verbose=True)

    # --- verifica bit-accurate sul test ------------------------------------
    Xte = np.load(cache / "test_x.npy", mmap_mode="r")
    Yte = np.load(cache / "test_y.npy")
    YMte = np.load(cache / "test_ym.npy").astype(np.int64)
    n = len(Yte) if N_EVAL_FX is None else min(N_EVAL_FX, len(Yte))
    blocks = [(i, min(i + CHUNK_FX, n)) for i in range(0, n, CHUNK_FX)]
    print(f"\nverifica bit-accurate su {n} record di test, {WORKERS_FX} processi ...")
    _CTX.update(E=E, G=G, B=B, X=Xte, QZ=QZ, QA=QA, NF=nf_qa, enc=enc, luts=luts)
    t0 = time.time()
    xbar, sat = run_chain_parallel(blocks)
    print(f"  {time.time() - t0:.0f} s")

    # --- classificatore -----------------------------------------------------
    xbar_max = float(np.abs(xbar).max()) / 2.0 ** G.F_ACT
    f_cin_free = int(math.floor(math.log2(127 / max(xbar_max, 1e-12))))
    f_cin = G.F_ACT - HW["RQ_SHIFT"] if HW_LOCK else f_cin_free
    dec = E.write_decoder_mem(None, None, W_fix["cls_w"], W_fix["cls_b"], f_cin,
                              mem_dir, nlayer=T.N_LAYERS, verbose=False, with_norm=False)
    xq = E.satn(E.shr_r(xbar, dec["rq_shift"]), E.W_CIN)
    sat_cin = int((np.abs(E.shr_r(xbar, dec["rq_shift"])) > 127).sum())
    logits_fx = E.classify_fx(xbar, dec)
    print(f"classificatore: F_CIN {f_cin} (libero {f_cin_free}), RQ_SHIFT {dec['rq_shift']}, "
          f"F_W {dec['f_w']}, max|xbar| {xbar_max:.3f}, ingressi INT8 tosati {sat_cin}")

    # --- riferimento float sugli stessi record ------------------------------
    lf = []
    with torch.no_grad():
        for i in range(0, n, 64):
            x = torch.from_numpy(np.asarray(Xte[i:min(i + 64, n)], dtype=np.float32)
                                 / 2 ** T.F_IN).to(dev)
            lf.append(model(x).double().cpu().numpy())
    logits_f = np.concatenate(lf)

    scaled = logits_fx / 2.0 ** (dec["f_w"] + dec["f_cin"])
    err = scaled - logits_f
    snr = 20 * math.log10(np.sqrt((logits_f ** 2).mean()) / max(np.sqrt((err ** 2).mean()), 1e-12))
    mf = metrics(logits_f, YMte[:n], Yte[:n])
    mq = metrics(logits_fx.astype(np.float64), YMte[:n], Yte[:n])
    agree = float((logits_fx.argmax(1) == logits_f.argmax(1)).mean())

    tot_sat = (sat["enc"] + sat["final"] + sat_cin
               + sum(l["sat_x"] + l["sat_a"] + l["sat_y"] + l["norm"] for l in sat["layers"]))
    print(f"\nSNR logit {snr:.1f} dB   argmax concordi {agree:.4f}   saturazioni totali {tot_sat}")
    print(f"  {'layer':>6s}{'sat_x':>10s}{'sat_a':>10s}{'sat_y':>10s}{'norm':>10s}")
    for m, l in enumerate(sat["layers"]):
        print(f"  {m:>6d}{l['sat_x']:>10d}{l['sat_a']:>10d}{l['sat_y']:>10d}{l['norm']:>10d}")
    print(f"  encoder {sat['enc']}   norm finale {sat['final']}   ingresso INT8 {sat_cin}")

    tf = auc_table(f"PTB-XL superdiagnostic, fold 10 -- FLOAT ({n} record)", YMte[:n], mf)
    tq = auc_table(f"PTB-XL superdiagnostic, fold 10 -- FIXED bit-accurate "
                   f"({'generic del bitstream' if HW_LOCK else 'generic liberi'}, {n} record)",
                   YMte[:n], mq)
    summary = (f"{tf}\n\n{tq}\n\n"
               f"accuracy etichetta singola ({mf['n_single']} record, argmax):  "
               f"float {mf['acc']:.4f}   fixed {mq['acc']:.4f}   "
               f"(drop {100 * (mf['acc'] - mq['acc']):+.2f} punti)\n"
               f"balanced accuracy: float {mf['bal_acc']:.4f}   fixed {mq['bal_acc']:.4f}\n"
               f"SNR logit {snr:.1f} dB   argmax concordi {agree:.4f}   saturazioni {tot_sat}")
    print("\n" + summary)

    # --- report -------------------------------------------------------------
    # input_scale: SCALE della scala fissa sugli ingressi (1/mV), letta da
    # export_board_files.py per generare gli stimoli della scheda. Da non
    # confondere con `scale`, la riscalatura del residuo.
    meta_path = cache / "meta.json"
    input_scale = json.loads(meta_path.read_text()).get("scale") if meta_path.is_file() else None
    rep = dict(
        checkpoint=str(CKPT), hw_lock=HW_LOCK, hw_generics=HW, n_cal=n_cal, n_eval_fx=n,
        norm_mode=rep_train["norm_mode"], input_scale=input_scale,
        scale=s, fill=FILL, f_ssm=f_ssm, f_ssm_free=f_ssm_free, f_d=F_D, f_d_calibrati=f_d_cal,
        peaks_before={k: rng_before[f"max_{k}"] for k in ("x", "a", "yssm")},
        peaks_after={k: rng_after[f"max_{k}"] for k in ("x", "a", "yssm")},
        per_layer_x_before=rng_before["x"], per_layer_x_after=rng_after["x"],
        encoder=dict(F_WENC=enc["f_enc"], F_WENC_free=enc["f_free"], SH_OUT=enc["sh_out"],
                     sat_w=enc["sat_w"], sat_b=enc["sat_b"], max_w=enc["max_w"], max_b=enc["max_b"]),
        decoder=dict(F_CIN=dec["f_cin"], F_CIN_free=f_cin_free, F_W=dec["f_w"],
                     RQ_SHIFT=dec["rq_shift"], xbar_max=xbar_max, sat_cin=sat_cin),
        # `affine[m].SH` e' il campo che Export_s4d_wstream.py legge per ld_bn_sh
        affine=[dict(idx=m, F_BNW=q["f_bnw"], F_B=q["f_b"], SH=q["sh"], max_w=q["max_w"],
                     max_b=q["max_b"], rel_worst=q["rel_worst"], warn=q["warn"])
                for m, q in enumerate(QA + [nf_qa])],
        layers=[dict(jury_before=q["jury_before"], jury_after=q["jury_after"], rho=q["rho"],
                     f_d=q["f_d"], f_d_cal=q["f_d_cal"], bsh_min=int(q["bsh"].min()),
                     bsh_max=int(q["bsh"].max()), warn=q["warn"], **sat["layers"][m])
                for m, q in enumerate(QZ)],
        sat_total=tot_sat, sat_enc=sat["enc"], sat_final=sat["final"],
        logit_snr_db=round(snr, 2), agreement=agree, float=mf, fixed=mq,
    )
    (out_dir / "quant_report.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "summary.txt").write_text(summary + "\n")
    if WRITE_MEM:
        shutil.copy2(out_dir / "quant_report.json", mem_dir / "quant_report.json")
        shutil.copy2(Path(__file__), out_dir / Path(__file__).name)
    else:
        shutil.rmtree(mem_dir, ignore_errors=True)
    print(f"\noutput: {out_dir}" + (f"   .mem: {mem_dir}" if WRITE_MEM else ""))


if __name__ == "__main__":
    main()
