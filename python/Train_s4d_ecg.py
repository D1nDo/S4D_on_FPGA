#!/usr/bin/env python3
"""
Train_s4d_ecg.py -- addestramento S4D su PTB-XL (ECG a 12 derivazioni),
5 superclassi diagnostiche, per il core S4D con L = 1024.

Si lancia senza argomenti, dalla radice del repository:

    conda run -n mamba_env python Train_s4d_ecg.py

Tutte le manopole sono costanti nel blocco CONFIGURAZIONE: come per
Train_s4d_speech_command_8bit.py, l'esperimento e' documentato dal file, non
da una riga di comando.

PERCHE' UN NUOVO TRAINING
Il modello ECG precedente (paper_S4D/ECG/mem/) e' stato addestrato su L = 2500
passi (500 Hz, stride 2, POOL_DIVISOR = 2500), ma il core in scheda ha L = 1024
e LOG2_L = 10: Export_ECG_stimulus.py gli manda solo i primi 1024 passi, cioe'
il 41% del segnale visto in training, e il mean pool divide per 1024 invece
che per 2500. Qui si addestra direttamente sulla sequenza che l'hardware vede.

DATI: 100 Hz, 10 s INTERI
records100/ di PTB-XL: 1000 campioni x 12 derivazioni, gia' ricampionati da
PhysioNet. 1000 passi entrano in L = 1024 con 24 zeri di coda, quindi la rete
vede l'intero record invece di 4.1 s. Il padding viene DOPO la normalizzazione,
cosi' la coda e' zero esatto; il mean pool somma anche quei 24 passi e divide
per 1024, esattamente come s4d_meanpool.v (>>> LOG2_L).

Preelaborazione, identica a quella di Export_ECG_stimulus.py tranne la
frequenza (e che per la scheda dovra' essere la STESSA di preprocess() qui):
  1. adu int16 / 1000 -> mV
  2. z-score per record e per derivazione sui 1000 campioni
  3. Q5.11: round(z * 2048), saturazione int16  (formato d'ingresso di
     s4d_encoder_ecg_axis.v: signed, F_ACT = 11)
  4. padding con zeri fino a 1024
Il modello riceve q / 2048, cioe' esattamente il valore fixed point: niente
MU/SIGMA da ripiegare nell'encoder, la normalizzazione e' gia' nel dato.

ETICHETTE
Superclasse diagnostica (CD, HYP, MI, NORM, STTC, ordine alfabetico come in
Export_ECG_stimulus.py), da tutti i codici SCP con diagnostic == 1 a qualsiasi
likelihood, tenendo solo i record con UNA sola superclasse. Con questa regola
il fold 10 coincide record per record ed etichetta per etichetta con
data_ptbxl/fold10_single.csv (verificato: 1650 record). Split raccomandato
dagli autori: fold 1-8 training, 9 validazione, 10 test.

CLASSI SBILANCIATE
NORM e' ~55% del test, HYP ~3%. Oltre all'accuracy si registrano balanced
accuracy e macro-F1: un modello che dice sempre NORM fa 55% di accuracy.
CLASS_WEIGHTING = "inv_sqrt" pesa la cross-entropy con 1/sqrt(frequenza), un
compromesso: "inv" puro sacrifica accuracy globale per le classi rare.
"""

import ast
import copy
import datetime
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# CONFIGURAZIONE -- si modifica qui, non da riga di comando
# ============================================================================

DATA = Path("data_ptbxl")                   # ptbxl_database.csv, records100/
CACHE_DIR = Path("data/ptbxl_cache")        # .npy int16 preelaborati
EXP_ROOT = Path("experiments")

# --- normalizzazione dell'ingresso ------------------------------------------
# "zscore": (x - media) / std per record e per derivazione (run A). Cancella
#           l'ampiezza assoluta e i rapporti di ampiezza fra derivazioni.
# "fixed" : (x - mediana per derivazione) * SCALE, con SCALE UNICA per tutto il
#           dataset: le ampiezze restano confrontabili fra pazienti e fra
#           derivazioni (criteri di voltaggio per HYP, onde Q per MI).
NORM_MODE = "fixed"
FIXED_TARGET = 4.0      # SCALE porta il percentile FIXED_PCT di |x| a questo
FIXED_PCT = 99.9        # valore (Q5.11 satura a +-16: margine di 4x)
FIXED_CAL_N = 2000      # record del training per misurare SCALE

# --- dati -------------------------------------------------------------------
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]
NLEAD = 12
# FS = 100: records100, 1000 campioni + 24 zeri = L del core (LOG2_L = 10).
# FS = 500: records500, 5000 campioni, record intero a piena risoluzione. E' un
#           LIMITE SUPERIORE in float: con L = 5000 il modello NON gira sul
#           core attuale (L = 1024 fisso) e non va esportato.
FS = 100
RAW_L = 10 * FS
L = 1024 if FS == 100 else RAW_L
REC_COL = "filename_lr" if FS == 100 else "filename_hr"
HW_COMPATIBLE = (L == 1024)
F_IN = 11           # Q5.11, F_ACT dell'encoder
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLDS = (9,)
TEST_FOLDS = (10,)

# --- modello: i default tengono la compatibilita' con l'RTL (H=128, MODES=32)
D_MODEL = 128
N_LAYERS = 6
D_STATE = 64
DROPOUT = 0.1

# --- ottimizzazione ---------------------------------------------------------
LR = 1e-2
LR_SSM = 1e-3       # gruppo del kernel S4D, weight decay 0
WD = 0.05
BATCH = 32
EPOCHS = 40
WARMUP_EPOCHS = 2
PATIENCE = 10
WORKERS = 4
SEED = 1337
# LABEL_MODE = "single": solo record con UNA superclasse, softmax + cross-entropy
#              (run A, B). Ottimizza l'accuracy a etichetta singola.
# LABEL_MODE = "multi" : tutti i record con >= 1 superclasse, etichette multi-hot,
#              sigmoide per classe + BCE: il setup del benchmark PTB-XL, che
#              ottimizza la macro-AUC. In scheda non cambia niente: 5 logit, argmax.
LABEL_MODE = "multi"
CLASS_WEIGHTING = None          # solo LABEL_MODE="single": None | "inv_sqrt" | "inv"
SELECT_ON = "macro_auc"         # metrica di validazione per il best:
                                # "macro_auc" | "acc" | "bal_acc"

# --- diagnostica ------------------------------------------------------------
PRINT_SUMMARY = True
SUMMARY_ONLY = False    # True: stampa il sommario ed esce senza addestrare
SUBSET = None           # int per una prova rapida, None per il dataset intero


# ============================================================================
# moduli del progetto
# ============================================================================
def load_modules():
    """gen_s4d_mem / s4d_export / s4d_bn: i modelli golden bit-accurate.

    Sono file separati in questa cartella e sono il contratto con l'RTL
    (formati fixed point, clamp di Jury, packing dei .mem).
    """
    import gen_s4d_mem as G, s4d_export as E, s4d_bn as B
    return G, E, B


def ecg_model(B, n_classes):
    """Il modello BN di s4d_bn con l'encoder a NLEAD ingressi.

    Si sostituisce solo input_proj: nucleo S4D, norm, GLU e classificatore
    restano l'oggetto che la quantizzazione sa esportare. s4d_encoder_ecg_axis.v fa
    y[ch] = sum_c W[ch][c] * x[c] + b[ch], cioe' proprio un Linear(12, H).
    """
    m = B.torch_model(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                      dropout=DROPOUT, n_classes=n_classes)
    m.input_proj = nn.Linear(NLEAD, D_MODEL)
    return m


# ============================================================================
# dati
# ============================================================================
def read_mv(path):
    """Un record records{FS} -> (RAW_L, NLEAD) float64 in mV."""
    x = np.fromfile(path, dtype="<i2")
    assert x.size == RAW_L * NLEAD, f"{path}: {x.size} campioni"
    return x.reshape(RAW_L, NLEAD).astype(np.float64) / 1000.0


def normalize(x, mode=NORM_MODE, scale=None):
    """(RAW_L, NLEAD) mV -> valori reali prima della quantizzazione Q5.11."""
    if mode == "zscore":
        sd = x.std(0)
        return (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)
    if mode == "fixed":
        # la mediana toglie la baseline senza farsi tirare dai QRS come la media
        return (x - np.median(x, axis=0)) * scale
    raise ValueError(f"NORM_MODE sconosciuto: {mode}")


def preprocess(path, mode=NORM_MODE, scale=None):
    """Un record records{FS} -> ((L, NLEAD) int16 Q5.11 con padding a L,
    numero di campioni saturati).

    E' la funzione da usare anche per generare gli stimoli della scheda: il
    codice intero restituito e' quello che l'encoder riceve.
    """
    z = normalize(read_mv(path), mode, scale)
    r = np.round(z * (1 << F_IN))
    n_sat = int((np.abs(r) > 32767).sum())
    q = np.clip(r, -32768, 32767).astype(np.int16)
    out = np.zeros((L, NLEAD), dtype=np.int16)
    out[:RAW_L] = q
    return out, n_sat


def calibrate_scale(files, data=DATA, n=FIXED_CAL_N, pct=FIXED_PCT,
                    target=FIXED_TARGET):
    """SCALE unica, misurata SOLO sul training: il percentile pct di
    |x - mediana| (su tutte le derivazioni) va a `target`."""
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    a = np.concatenate([np.abs(normalize(read_mv(data / f"{files[i]}.dat"),
                                         "fixed", 1.0)).ravel() for i in pick])
    p = float(np.percentile(a, pct))
    print(f"  scala fissa: p{pct} di |x - mediana| = {p:.3f} mV "
          f"-> SCALE {target / p:.4f} /mV  (fondo scala Q5.11 = "
          f"+-{16 * p / target:.1f} mV)")
    return target / p


def superclass_table(data=DATA, label_mode=LABEL_MODE):
    """ecg_id -> strat_fold, file, label (superclasse se singola, altrimenti
    None) e una colonna 0/1 per classe. "single" tiene i record con UNA
    superclasse, "multi" quelli con almeno una. Regola che riproduce
    fold10_single.csv (vedi docstring)."""
    import pandas as pd

    db = pd.read_csv(data / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db.scp_codes.apply(ast.literal_eval)
    ag = pd.read_csv(data / "scp_statements.csv", index_col=0)
    ag = ag[ag.diagnostic == 1]
    sup = db.scp_codes.apply(
        lambda d: sorted({ag.loc[k].diagnostic_class for k in d if k in ag.index}))
    n = sup.apply(len)
    db = db[(n == 1) if label_mode == "single" else (n >= 1)].copy()
    db["label"] = sup[db.index].apply(lambda s: s[0] if len(s) == 1 else None)
    for c in CLASSES:
        db[c] = sup[db.index].apply(lambda s, c=c: int(c in s))

    ref_path = data / "fold10_single.csv"
    if ref_path.is_file():
        ref = pd.read_csv(ref_path, index_col="ecg_id")
        f10 = db[(db.strat_fold == 10) & db.label.notna()]
        assert set(f10.index) == set(ref.index) and \
            (f10.loc[ref.index, "label"] == ref.label).all(), \
            "il fold 10 non coincide con fold10_single.csv: regola delle etichette cambiata?"
    return db[["strat_fold", REC_COL, "label"] + CLASSES].rename(columns={REC_COL: "file"})


def build_cache(data=DATA, mode=NORM_MODE):
    """Preelabora i record una volta sola: {split}_x.npy (N, L, 12) int16.

    Una cache per normalizzazione (la z-score del run A resta in CACHE_DIR):
    i due run si confrontano sugli stessi record. meta.json conserva SCALE e i
    campioni saturati, che servono poi a quantizzatore e stimoli della scheda.
    """
    if FS == 100:       # nomi del run A / run B, invariati
        cache_dir = CACHE_DIR if mode == "zscore" else \
            CACHE_DIR.with_name(f"{CACHE_DIR.name}_{mode}")
    else:
        cache_dir = CACHE_DIR.with_name(f"{CACHE_DIR.name}_fs{FS}_{mode}")
    if LABEL_MODE == "multi":   # insieme di record diverso: cache separata
        cache_dir = cache_dir.with_name(cache_dir.name + "_multi")
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else \
        dict(norm_mode=mode, sat={})
    tab = None
    for split, folds in (("train", TRAIN_FOLDS), ("val", VAL_FOLDS), ("test", TEST_FOLDS)):
        fx, fy, fi, fm = (cache_dir / f"{split}_{s}.npy" for s in ("x", "y", "id", "ym"))
        if fx.is_file() and fy.is_file() and fi.is_file() and \
                (LABEL_MODE == "single" or fm.is_file()):
            continue
        if tab is None:
            tab = superclass_table(data)
            missing = [f for f in tab.file
                       if not (data / f"{f}.dat").is_file()]
            if missing:
                sys.exit(f"{len(missing)} record di records{FS} mancanti "
                         f"(es. {data / missing[0]}.dat): scaricare PTB-XL 1.0.3")
            if mode == "fixed" and "scale" not in meta:
                tr_files = tab[tab.strat_fold.isin(TRAIN_FOLDS)].file.to_numpy()
                meta["scale"] = calibrate_scale(tr_files, data)
        sel = tab[tab.strat_fold.isin(folds)]
        print(f"  preelaborazione {split}: {len(sel)} record ...")
        out = [preprocess(data / f"{f}.dat", mode, meta.get("scale"))
               for f in sel.file]
        X = np.stack([o[0] for o in out])
        n_sat = sum(o[1] for o in out)
        meta["sat"][split] = dict(samples=n_sat,
                                  frac=n_sat / (len(sel) * RAW_L * NLEAD))
        print(f"    campioni saturati in Q5.11: {n_sat} "
              f"({100 * meta['sat'][split]['frac']:.4f}%)")
        # y: indice della superclasse se singola, -1 se il record ne ha piu' di una
        Y = np.array([CLASSES.index(c) if c is not None else -1 for c in sel.label],
                     dtype=np.int64)
        np.save(fx, X); np.save(fy, Y); np.save(fi, sel.index.to_numpy())
        np.save(fm, sel[CLASSES].to_numpy(dtype=np.int8))
        meta_path.write_text(json.dumps(meta, indent=2))
    return {s: (cache_dir / f"{s}_x.npy", cache_dir / f"{s}_y.npy",
                cache_dir / f"{s}_ym.npy") for s in ("train", "val", "test")}, meta


class ECGDataset(torch.utils.data.Dataset):
    """(L, 12) float = codice Q5.11 / 2048: il valore che l'encoder vede.
    Restituisce (x, etichette multi-hot, superclasse singola o -1)."""

    def __init__(self, x_path, y_path, ym_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path)
        self.YM = np.load(ym_path).astype(np.float32) if Path(ym_path).is_file() \
            else np.eye(len(CLASSES), dtype=np.float32)[self.Y]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        x = np.asarray(self.X[i], dtype=np.float32) / (1 << F_IN)
        return torch.from_numpy(x), torch.from_numpy(self.YM[i]), int(self.Y[i])


def build_loaders():
    paths, meta = build_cache()

    def mk(split, shuffle):
        ds = ECGDataset(*paths[split])
        if SUBSET:
            n = min(SUBSET if split == "train" else max(SUBSET // 8, 1), len(ds))
            ds = torch.utils.data.Subset(ds, range(n))
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH, shuffle=shuffle, drop_last=shuffle,
            num_workers=WORKERS, pin_memory=True)

    loaders = mk("train", True), mk("val", False), mk("test", False)
    counts = {}
    for s in paths:
        ys = np.load(paths[s][1])
        c = np.bincount(ys[ys >= 0], minlength=len(CLASSES))
        pos = loaders[("train", "val", "test").index(s)].dataset
        pos = (pos.dataset if isinstance(pos, torch.utils.data.Subset) else pos).YM.sum(0)
        counts[s] = c
        print(f"  {s:5s} {len(ys):6d} record, {int(c.sum())} a etichetta singola.  "
              "positivi: " + "  ".join(f"{k} {int(v)}" for k, v in zip(CLASSES, pos)))
    return loaders, counts, meta


def class_weights(counts, mode=CLASS_WEIGHTING):
    if mode is None:
        return None
    f = counts / counts.sum()
    w = 1.0 / np.sqrt(f) if mode == "inv_sqrt" else 1.0 / f
    w = w / (w * f).sum()           # peso medio 1 sulla distribuzione del training
    return torch.tensor(w, dtype=torch.float32)


# ============================================================================
# scheduler
# ============================================================================
def cosine_with_warmup(opt, warmup_steps, total_steps, min_ratio=0.0):
    def f(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ============================================================================
# metriche
# ============================================================================
def auc(y, s):
    """AUC binaria con la statistica di Mann-Whitney (pari merito: rango medio)."""
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, r) / cnt)[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def evaluate(model, loader, device):
    """Macro-AUC su tutti i record (etichette multi-hot, logit come punteggi),
    e accuracy / balanced accuracy / macro-F1 / matrice di confusione con
    l'argmax sui record a etichetta singola, cioe' quello che fa la scheda."""
    model.eval()
    k = len(CLASSES)
    Z, YM, Y = [], [], []
    with torch.no_grad():
        for x, ym, y in loader:
            Z.append(model(x.to(device)).float().cpu().numpy())
            YM.append(ym.numpy()); Y.append(y.numpy())
    Z, YM, Y = np.concatenate(Z), np.concatenate(YM), np.concatenate(Y)
    aucs = [auc(YM[:, c], Z[:, c]) for c in range(k)]
    one = Y >= 0
    cm = np.zeros((k, k), dtype=np.int64)          # righe vero, colonne predetto
    np.add.at(cm, (Y[one], Z[one].argmax(-1)), 1)
    tp = np.diag(cm).astype(np.float64)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return dict(acc=float(tp.sum() / max(cm.sum(), 1)),
                bal_acc=float(rec.mean()), macro_f1=float(f1.mean()),
                recall={c: float(r) for c, r in zip(CLASSES, rec)},
                cm=cm.tolist(), macro_auc=float(np.nanmean(aucs)),
                auc={c: a for c, a in zip(CLASSES, aucs)},
                n_records=int(len(Y)), n_single=int(one.sum()))


def print_summary(model):
    print("\n===== ARCHITETTURA =====")
    print(model)
    print("\n===== PARAMETRI =====")
    tot = 0
    for name, p in model.named_parameters():
        tot += p.numel()
        print(f"{name:52s} {str(tuple(p.shape)):20s} {p.numel():>9,}")
    print(f"\nPARAMETRI TOTALI: {tot:,}")
    print("=" * 60 + "\n")
    return tot


def param_groups(model):
    """Il kernel S4D fuori dal weight decay, come per Speech Commands."""
    ssm_keys = ("kernel.log_A_real", "kernel.A_imag", "kernel.log_dt")
    ssm, nod, dec = [], [], []
    for n_, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        if any(k in n_ for k in ssm_keys):
            ssm.append(p_)
        elif n_.endswith(".bias") or ".bn." in n_ or n_.endswith(".D"):
            nod.append(p_)
        else:
            dec.append(p_)
    assert len(ssm) == 3 * N_LAYERS, f"filtro SSM: {len(ssm)}, attesi {3*N_LAYERS}"
    assert nod, "filtro no_decay vuoto: nomi di BatchNormSeq cambiati?"
    return dec, nod, ssm


# ============================================================================
# addestramento
# ============================================================================
def train(model, loaders, dev, exp_dir, weights):
    tr, va, te = loaders
    dec, nod, ssm = param_groups(model)
    print(f"parametri: {len(dec)} con WD, {len(nod)} senza, {len(ssm)} SSM")

    opt = torch.optim.AdamW([
        {"params": dec, "weight_decay": WD},
        {"params": nod, "weight_decay": 0.0},
        {"params": ssm, "weight_decay": 0.0, "lr": LR_SSM},
    ], lr=LR)
    spe = max(len(tr), 1)
    sched = cosine_with_warmup(opt, WARMUP_EPOCHS * spe, EPOCHS * spe)
    if LABEL_MODE == "multi":
        bce = nn.BCEWithLogitsLoss()
        crit = lambda z, ym, y: bce(z, ym)
    else:
        ce = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(dev))
        crit = lambda z, ym, y: ce(z, y)
    log_path = exp_dir / "train_log.jsonl"

    best, best_sd, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(EPOCHS):
        model.train(); tot = 0.0; t0 = time.time(); nan_hit = False
        for x, ym, y in tr:
            x, ym, y = (t.to(dev, non_blocking=True) for t in (x, ym, y))
            loss = crit(model(x), ym, y)
            # float32 sempre: in half il kernel S4D va in nan (vedi lo script
            # Speech Commands). Un nan qui e' il forward, non il gradiente.
            if not torch.isfinite(loss):
                nan_hit = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if nan_hit:
            print(f"loss non finita all'epoca {ep+1}: interrotto (abbassare LR). "
                  f"Il modello migliore resta in {exp_dir/'best.pt'}")
            break

        m = evaluate(model, va, dev)
        dt = time.time() - t0
        avg = tot / spe
        print(f"epoca {ep+1}/{EPOCHS}  loss {avg:.4f}  val AUC {m['macro_auc']:.4f}  "
              f"acc {m['acc']:.4f}  "
              f"bal {m['bal_acc']:.4f}  F1 {m['macro_f1']:.4f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}  ({dt:.0f}s)")
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(epoch=ep + 1, loss=avg,
                                    val_macro_auc=m["macro_auc"], val_acc=m["acc"],
                                    val_bal_acc=m["bal_acc"],
                                    val_macro_f1=m["macro_f1"],
                                    lr=opt.param_groups[0]["lr"],
                                    sec=round(dt, 1))) + "\n")

        score = m[SELECT_ON]
        if score > best:
            best, best_sd, bad = score, copy.deepcopy(model.state_dict()), 0
            torch.save(best_sd, exp_dir / "best.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping (best val {SELECT_ON} {best:.4f})")
                break
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep, best=best,
                        patience=bad), exp_dir / "last.pt")

    model.load_state_dict(best_sd)
    return model, best


# ============================================================================
# main
# ============================================================================
def main():
    G, E, B = load_modules()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("device:", dev)

    if D_MODEL != G.H:
        print(f"ATTENZIONE: D_MODEL={D_MODEL} ma l'RTL ha H={G.H}: non esportabile.")
    if D_STATE // 2 != G.MODES:
        print(f"ATTENZIONE: D_STATE/2={D_STATE//2} ma l'RTL ha MODES={G.MODES}.")
    if not HW_COMPATIBLE:
        print(f"ATTENZIONE: FS={FS} Hz, L={L}: esperimento di riferimento in float. "
              f"Il core ha L=1024 fisso: questo modello NON e' esportabile.")

    loaders, counts, meta = build_loaders()
    print(f"normalizzazione: {NORM_MODE}" +
          (f"  SCALE {meta['scale']:.4f} /mV" if "scale" in meta else ""))
    weights = class_weights(counts["train"])
    if weights is not None:
        print("pesi della loss:", "  ".join(f"{c} {w:.2f}"
                                            for c, w in zip(CLASSES, weights.tolist())))

    model = ecg_model(B, len(CLASSES)).to(dev)
    n_par = print_summary(model) if PRINT_SUMMARY else None
    if SUMMARY_ONLY:
        print("SUMMARY_ONLY attivo: niente addestramento.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    exp_dir = EXP_ROOT / f"exp_{stamp}_ecg{FS}_l{L}_{NORM_MODE}_{LABEL_MODE}"
    exp_dir.mkdir(parents=True, exist_ok=False)     # mai sovrascrivere un run
    shutil.copy2(__file__, exp_dir / Path(__file__).name)
    print("output:", exp_dir)

    model, best_val = train(model, loaders, dev, exp_dir, weights)
    val = evaluate(model, loaders[1], dev)
    test = evaluate(model, loaders[2], dev)
    print(f"\ntest: macro-AUC {test['macro_auc']:.4f} ({test['n_records']} record)   "
          + "  ".join(f"{c} {a:.3f}" for c, a in test["auc"].items()))
    print(f"test a etichetta singola ({test['n_single']} record, argmax come in scheda): "
          f"acc {test['acc']:.4f}  bal {test['bal_acc']:.4f}  "
          f"F1 {test['macro_f1']:.4f}   (best val {SELECT_ON} {best_val:.4f})")
    print("recall per classe:", "  ".join(f"{c} {r:.3f}" for c, r in test["recall"].items()))

    (exp_dir / "report.json").write_text(json.dumps(dict(
        task=f"ptbxl_superdiag_{LABEL_MODE}", label_mode=LABEL_MODE, classes=CLASSES,
        input=f"records{FS} {NORM_MODE} Q5.11", fs=FS, hw_compatible=HW_COMPATIBLE, norm_mode=NORM_MODE,
        scale=meta.get("scale"), fixed_target=FIXED_TARGET, fixed_pct=FIXED_PCT,
        sat=meta.get("sat"),
        raw_len=RAW_L, seq_len=L, n_lead=NLEAD, f_in=F_IN,
        folds=dict(train=TRAIN_FOLDS, val=VAL_FOLDS, test=TEST_FOLDS),
        counts={s: c.tolist() for s, c in counts.items()},
        val=val, test=test, select_on=SELECT_ON, best_val=best_val,
        n_params=n_par, seed=SEED, d_model=D_MODEL, n_layers=N_LAYERS,
        d_state=D_STATE, dropout=DROPOUT, lr=LR, lr_ssm=LR_SSM, wd=WD,
        batch=BATCH, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE, class_weighting=CLASS_WEIGHTING,
        subset=SUBSET), indent=2))
    print("report:", exp_dir / "report.json")

    if not HW_COMPATIBLE:
        print(f"\nRiferimento float a {FS} Hz, L={L}: nessun export sull'acceleratore.")
        return
    print("\n--- note per l'export sull'acceleratore ---")
    print("1. K_CLS = 5; encoder a 12 ingressi (s4d_encoder_ecg_axis.v, Q5.11 con segno):")
    print("   il quantizzatore deve esportare W (128x12) e b, non il w/b a 1 canale.")
    print(f"2. Niente MU/SIGMA: la normalizzazione ({NORM_MODE}) e' gia' nello stimolo"
          + (f", SCALE {meta['scale']:.4f} /mV." if "scale" in meta else "."))
    print("3. Stimoli in scheda: generarli con preprocess() di questo file, stessa")
    print("   NORM_MODE e stessa SCALE (records100, 1000 campioni + 24 zeri),")
    print("   NON con la versione a 500 Hz di Export_ECG_stimulus.py.")


if __name__ == "__main__":
    main()
