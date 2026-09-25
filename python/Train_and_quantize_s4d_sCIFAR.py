"""
train_and_quantize_bn.py -- addestramento da zero + quantizzazione BatchNorm
per l'acceleratore S4D. Standalone: include gen_s4d_mem, s4d_export, s4d_bn.
"""

from pathlib import Path
import sys, os, json, math, time, shutil, types

FILL      = 0.75
N_CAL     = 128
N_EVAL_FX = 10000      # era 32: con 32 immagini l'errore standard su acc_fx e' ~+-9 punti
CHUNK_FX  = 64       # blocchi della verifica fixed; vedi il commento dove viene usato
SEED      = 1337
MAX_EPOCHS   = 100
TRAIN_SUBSET = None
BATCH = 64
PATIENCE = 10
 
REPO = WORK = Path(__file__).resolve().parent
DATA_ROOT = REPO / "data"
 
MEM_OUT  = WORK / "mem_trained_bn"
CKPT_DIR = WORK / "ckpt"
LOG_DIR  = WORK / "logs"
for d in (MEM_OUT, CKPT_DIR, LOG_DIR, DATA_ROOT):
    d.mkdir(parents=True, exist_ok=True)
 
BEST_CKPT = CKPT_DIR / "s4d_scifar_configC_bn_best.pt"
LAST_CKPT = CKPT_DIR / "last_bn.pt"
RUN_LOG   = LOG_DIR / "train_log_bn.jsonl"
 

# ============================================================================
# modelli golden bit-accurate: gen_s4d_mem / s4d_export / s4d_bn
# ============================================================================
# Sono file separati in questa cartella (prima erano incorporati qui come
# stringhe). Contengono il contratto con l'RTL: formati fixed point, clamp di
# Jury, packing dei .mem. Ordine degli import = ordine delle dipendenze.
import gen_s4d_mem as G
import s4d_export as E
import s4d_bn as B

# ============================================================================
# training + quantizzazione
# ============================================================================
 
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Subset
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print("device:", DEVICE)
 
N_LAYERS, SEQ_LEN = 6, 1024
model = B.torch_model().to(DEVICE)
 
MU, SIGMA = 0.4734, 0.2516
transform = T.Compose([T.Grayscale(1), T.ToTensor(), T.Normalize((MU,), (SIGMA,))])
train_full = torchvision.datasets.CIFAR10(str(DATA_ROOT), train=True,  download=True, transform=transform)
test_set   = torchvision.datasets.CIFAR10(str(DATA_ROOT), train=False, download=True, transform=transform)
 
n_tr = int(0.9 * len(train_full))
train_set, val_set = random_split(train_full, [n_tr, len(train_full) - n_tr],
                                  generator=torch.Generator().manual_seed(SEED))
if TRAIN_SUBSET:
    train_set = Subset(train_set, range(min(TRAIN_SUBSET, len(train_set))))
    val_set   = Subset(val_set,   range(min(TRAIN_SUBSET // 8, len(val_set))))
 
def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs.view(imgs.size(0), -1, 1), y
 
mk = lambda ds, sh: DataLoader(ds, batch_size=BATCH, shuffle=sh, drop_last=sh,
                               num_workers=2, collate_fn=collate)
train_loader, val_loader, test_loader = mk(train_set, True), mk(val_set, False), mk(test_set, False)
 
# Calibrazione dei range sul VALIDATION set, non sul test.
# I range misurati qui decidono la scala s e F_SSM: tararli sul test significa
# scegliere la quantizzazione guardando le stesse immagini su cui poi si misura
# l'accuratezza fixed point.
cal_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate)
print("train", len(train_set), "val", len(val_set), "test", len(test_set))
 
 
def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
 
 
def evaluate(loader, limit=None):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if limit and i * BATCH >= limit:
                break
            pred = model(x.to(DEVICE)).argmax(-1).cpu()
            ok += (pred == y).sum().item(); tot += y.numel()
    return ok / max(tot, 1)
 
 
# ---------------------------------------------------------------- optimizer
# Il kernel S4D non va decaduto.
#
#   A_imag      parte da pi*n, quindi fino a ~97. Un weight_decay 0.01 applicato
#               ~700 volte per epoca lo schiaccia verso zero: i modi ad alta
#               frequenza collassano e i 32 modi per canale servono a meno di
#               quello per cui sono stati messi.
#   log_A_real  parte da log(0.5); il decay lo porta verso 0, cioe' A_real -> 1,
#               cioe' poli spinti verso il cerchio unitario. E' un termine che
#               agisce direttamente su rho = |lambda_bar|, quindi su quanti
#               biquad il clamp di Jury deve correggere in export: si sta
#               regolarizzando la posizione dei poli senza volerlo.
#   log_dt      stessa storia sui passi di discretizzazione.
#
# Nel repo S4D di riferimento questi tre hanno weight_decay 0 e learning rate
# fisso, esclusi dallo scheduler.
#
# Fuori dal decay anche:
#   D           guadagno per canale, non una matrice. Decaderlo abbassa max|D| e
#               sposta la calibrazione di f_d senza nessuna giustificazione.
#   BatchNorm   gamma diventa w = gamma/sqrt(var+eps) in export; restringerlo
#               peggiora il rapporto quanto/|w|min di print_affine_table.
#   bias        prassi standard.
#
# Per l'ablazione: SSM_KEYS = () riproduce il comportamento vecchio (in quel
# caso va tolto anche il primo assert).
SSM_KEYS = ("kernel.log_A_real", "kernel.A_imag", "kernel.log_dt")
LR_SSM = 1e-3
 
ssm, no_decay, decay = [], [], []
for n_, p_ in model.named_parameters():
    if not p_.requires_grad:
        continue
    if any(k in n_ for k in SSM_KEYS):
        ssm.append(p_)
    elif n_.endswith(".bias") or ".bn." in n_ or n_.endswith(".D"):
        no_decay.append(p_)
    else:
        decay.append(p_)
 
# Un filtro che non trova niente e' silenzioso: senza questi assert, un rename
# dei sottomoduli fa ripartire l'addestramento identico a prima.
assert len(ssm) == 3 * N_LAYERS, f"filtro SSM: {len(ssm)} parametri, attesi {3*N_LAYERS}"
assert no_decay, "filtro no_decay vuoto: i nomi di BatchNormSeq sono cambiati?"
print(f"parametri: {len(decay)} con WD, {len(no_decay)} senza, {len(ssm)} SSM")
 
opt = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
    {"params": ssm,      "weight_decay": 0.0, "lr": LR_SSM},
], lr=1e-3)
SSM_GROUP = 2
 
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
crit = nn.CrossEntropyLoss()
best, patience, ep0 = 0.0, 0, 0
 
for ep in range(ep0, MAX_EPOCHS):
    model.train(); tot = 0.0; t0 = time.time()
    for x, y in train_loader:
        loss = crit(model(x.to(DEVICE)), y.to(DEVICE))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
    acc = evaluate(val_loader)
 
    # ReduceLROnPlateau scala TUTTI i gruppi: il gruppo SSM va riportato al suo
    # lr fisso subito dopo, altrimenti i poli smettono di muoversi molto prima
    # del resto della rete.
    sched.step(acc)
    opt.param_groups[SSM_GROUP]["lr"] = LR_SSM
 
    dt = time.time() - t0
    avg_loss = tot / max(len(train_loader), 1)
    line = dict(epoch=ep + 1, loss=avg_loss, val_acc=acc, sec=round(dt, 1),
                lr=opt.param_groups[0]["lr"], lr_ssm=opt.param_groups[SSM_GROUP]["lr"])
    print(f"epoca {ep+1}/{MAX_EPOCHS}  loss {avg_loss:.4f}  val_acc {acc:.4f}  ({dt:.0f}s)")
    if acc > best:
        best, patience = acc, 0
        save_atomic(model.state_dict(), BEST_CKPT)
    else:
        patience += 1
    save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(),
                     sched=sched.state_dict(), epoch=ep, best=best,
                     patience=patience), LAST_CKPT)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(line) + "\n")
    if patience >= PATIENCE:
        print("early stopping"); break
 
model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE, weights_only=True))
model.eval()
ACC_FP32 = evaluate(test_loader)
print(f"accuracy FP32 = {ACC_FP32:.4f}")
 
# --- quantizzazione ---
 
W = B.weights_to_numpy(model)
rng_before = E.calibrate_ranges(model, cal_loader, DEVICE, n_images=N_CAL)
E.print_ranges(rng_before)
 
peak_act = max(rng_before["max_x"], rng_before["max_a"])
s = E.choose_scale(peak_act, fill=FILL)
f_ssm = E.needed_f_ssm(rng_before["max_yssm"], fill=FILL)
E.set_f_ssm(f_ssm)
print(f"s = {s}   F_SSM = {f_ssm}")
 
W_fix = B.rescale_value_branch(W, s)
 
model_fix = B.torch_model().to(DEVICE)
B.numpy_to_torch_(model_fix, W_fix)
model_fix = model_fix.to(DEVICE).eval()
 
xb, _ = next(iter(cal_loader))
with torch.no_grad():
    l0 = model(xb.to(DEVICE)).double().cpu().numpy()
    l1 = model_fix(xb.to(DEVICE)).double().cpu().numpy()
d = np.abs(l1 - l0).max()
ulp = float(np.spacing(np.float32(max(np.abs(l0).max(), 1.0))))
assert d < 256 * ulp, "la riscalatura ha cambiato la funzione"
 
rng_after = E.calibrate_ranges(model_fix, cal_loader, DEVICE, n_images=N_CAL)
 
np.savez(CKPT_DIR / "weights_bn.npz",
         **{k: v for k, v in W_fix.items() if k != "layers"},
         **{f"l{m}_{k}": v for m, dd in enumerate(W_fix["layers"]) for k, v in dd.items()})
 
luts = E.write_luts(MEM_OUT, verbose=False)
os.remove(MEM_OUT / "isqrt_lut.mem")
 
f_d_cal = [G.calib(dd["D"], G.W_D) for dd in W_fix["layers"]]
F_D = min(f_d_cal)
print(f"F_D = {F_D}")
 
QZ, QA = [], []
for m, dd in enumerate(W_fix["layers"]):
    q = E.quantize_layer(dd, f_d=F_D)
    E.write_layer_mem(m, q, MEM_OUT, with_norm=False)
    qa = B.quantize_affine(dd["bn_w"], dd["bn_b"], name=f"l{m}_bn")
    B.write_affine_mem(m, qa, MEM_OUT)
    QZ.append(q); QA.append(qa)
 
enc = E.quantize_encoder(W_fix["enc_w"], W_fix["enc_b"], MU, SIGMA)
E.write_encoder_mem(enc, MEM_OUT)
 
nf_qa = B.write_decoder_affine_mem(W_fix["nf_w"], W_fix["nf_b"], MEM_OUT, nlayer=N_LAYERS)
 
B.print_affine_table(QA + [nf_qa], nlayer=N_LAYERS)
 
# --- verifica fixed point sul test set ---
 
def raw_pixels(ds, idx):
    out = []
    for i in idx:
        x, y = ds[i]
        pix = np.clip(np.round((x.numpy().reshape(-1) * SIGMA + MU) * 255.0), 0, 255)
        out.append((pix.astype(np.int64), y))
    return np.stack([o[0] for o in out]), np.array([o[1] for o in out])
 
 
idx = list(range(min(N_EVAL_FX, len(test_set))))
PIX, LBL = raw_pixels(test_set, idx)
 
norms_fx = [B.AffineFX(qa) for qa in QA]
nf_fx = B.AffineFX(nf_qa, w_out=G.W_ACT)
 
# La catena bit-accurate gira a blocchi: LayerFX alloca s1 e s2 di forma
# (B, H, MODES) int64 per ognuno dei sei layer, ~6 MB per immagine, e con
# N_EVAL_FX = 512 in un colpo solo la RAM non basta. Le sequenze sono
# indipendenti fra loro, quindi spezzare il batch non cambia un bit.
#
# norms_fx / nf_fx sono condivisi fra i blocchi e i loro contatori si accumulano
# da soli; i LayerFX invece li ricrea run_chain_fx a ogni chiamata, quindi
# sat_x / sat_a / sat_y vanno sommati a mano. Perderli non e' un dettaglio: sono
# l'unica prova che il modello addestrato sta dentro Q5.11.
sat_layers = [dict(sat_x=0, sat_a=0, sat_y=0, acc_bound=0) for _ in range(N_LAYERS)]
_parts = []
for _i in range(0, PIX.shape[0], CHUNK_FX):
    _xb, _lay = E.run_chain_fx(PIX[_i:_i + CHUNK_FX], QZ, enc, None, None, luts,
                               norms=norms_fx, norm_final=nf_fx)
    _parts.append(_xb)
    for _a, _l in zip(sat_layers, _lay):
        _a["sat_x"] += _l.sat_x
        _a["sat_a"] += _l.sat_a
        _a["sat_y"] += _l.sat_y
        _a["acc_bound"] = max(_a["acc_bound"], _l.acc_bound)
    print(f"  fixed {min(_i + CHUNK_FX, PIX.shape[0])}/{PIX.shape[0]}", end="\r")
print()
xbar = np.concatenate(_parts)
 
xbar_max = float(np.abs(xbar).max()) / 2.0 ** G.F_ACT
f_cin = int(math.floor(math.log2(127 / max(xbar_max, 1e-12))))
print(f"F_CIN = {f_cin}, RQ_SHIFT = {G.F_ACT - f_cin}")
 
dec = E.write_decoder_mem(None, None, W_fix["cls_w"], W_fix["cls_b"], f_cin,
                          MEM_OUT, nlayer=N_LAYERS, with_norm=False)
 
logits_fx = E.classify_fx(xbar, dec)
 
# anche il riferimento float a blocchi: 512 immagini da 1024 passi in un colpo
# solo non stanno in memoria GPU
_lf = []
with torch.no_grad():
    for _i in range(0, PIX.shape[0], CHUNK_FX):
        xin = torch.tensor((PIX[_i:_i + CHUNK_FX] / 255.0 - MU) / SIGMA,
                           dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        _lf.append(model(xin).double().cpu().numpy())
logits_f = np.concatenate(_lf)
 
scaled = logits_fx / 2.0 ** (dec["f_w"] + dec["f_cin"])
err = scaled - logits_f
snr = 20 * math.log10(np.sqrt((logits_f ** 2).mean()) / max(np.sqrt((err ** 2).mean()), 1e-12))
acc_fx = float((logits_fx.argmax(1) == LBL).mean())
acc_f = float((logits_f.argmax(1) == LBL).mean())
agree = float((logits_fx.argmax(1) == logits_f.argmax(1)).mean())
 
# Intervallo di confidenza: serve a non leggere come degradazione della
# quantizzazione quello che e' rumore campionario. Con 32 immagini valeva
# +-9 punti percentuali, cioe' il numero non diceva niente.
se95 = 1.96 * math.sqrt(max(acc_fx * (1 - acc_fx), 1e-12) / len(idx))
print(f"SNR logit {snr:.1f} dB   acc float {acc_f:.4f}   "
      f"acc fixed {acc_fx:.4f} +-{se95:.4f} (95%, n={len(idx)})   concordi {agree:.4f}")
 
tot_sat = (sum(a["sat_x"] + a["sat_a"] + a["sat_y"] for a in sat_layers)
           + sum(n.sat for n in norms_fx) + nf_fx.sat)
print(f"saturazioni totali: {tot_sat}")
 
# --- generic, stimoli testbench, bundle ---
 
QA_ALL = QA + [nf_qa]
gen_str = B.required_generics(enc, dec, QA_ALL, f_ssm=f_ssm, nlayer=N_LAYERS,
                          mem_dir=str(MEM_OUT) + "/", f_d=F_D)
(MEM_OUT / "top_generics.txt").write_text(gen_str + "\n")
 
NTB = min(2, len(idx))
E._write(str(MEM_OUT / "top_stim.mem"), [G.hx(int(v), 8) for v in PIX[:NTB].reshape(-1)])
E._write(str(MEM_OUT / "top_exp.mem"),
         [G.hx(int(v), E.W_ACC_D) for v in logits_fx[:NTB].reshape(-1)])
 
with open(MEM_OUT / "top_config.vh", "w") as f:
    for k, v in [("TB_L", PIX.shape[1]), ("TB_LOG2_L", int(math.log2(PIX.shape[1]))),
                 ("TB_H", E.H), ("TB_K", E.K_CLS), ("TB_NLAYER", N_LAYERS),
                 ("TB_IMGS", NTB), ("TB_NORM_AFFINE", 1),
                 ("TB_F_SSM", G.F_SSM), ("TB_F_OUT", G.F_ACT),
                 ("TB_RQ_SHIFT", dec["rq_shift"]), ("TB_F_D", F_D),
                 ("ENC_F_WENC", enc["f_enc"]), ("ENC_SH_OUT", enc["sh_out"]),
                 ("CLS_F_CIN", dec["f_cin"]), ("CLS_F_W", dec["f_w"])]:
        f.write(f"localparam integer {k} = {v};\n")
    f.write(f"localparam integer BN_W_W = {B.W_BNW};\n")
    f.write(f"localparam integer BN_W_B = {B.W_BNB};\n")
    f.write(f"localparam integer BN_W_ACC = {B.W_BNACC};\n")
    for m, qa in enumerate(QA_ALL):
        sh_val = qa["sh"]
        f.write(f"localparam integer BN_SH_{m} = {sh_val};\n")
    for m, qa in enumerate(QA_ALL):
        fbnw_val = qa["f_bnw"]
        f.write(f"localparam integer BN_F_W_{m} = {fbnw_val};\n")
    for m, qa in enumerate(QA_ALL):
        fb_val = qa["f_b"]
        f.write(f"localparam integer BN_F_B_{m} = {fb_val};\n")
 
report = dict(
    norm="batchnorm", seed=SEED, acc_fp32_test=ACC_FP32,
    scale=s, f_ssm=f_ssm, f_d=F_D, f_d_calibrati=f_d_cal,
    fill=FILL, n_cal=N_CAL, cal_split="val", n_eval_fx=len(idx),
    optimizer=dict(wd=0.01, lr=1e-3, lr_ssm=LR_SSM, ssm_keys=list(SSM_KEYS),
                   n_decay=len(decay), n_no_decay=len(no_decay), n_ssm=len(ssm)),
    peaks_before=dict(x=rng_before["max_x"], a=rng_before["max_a"], yssm=rng_before["max_yssm"]),
    peaks_after=dict(x=rng_after["max_x"], a=rng_after["max_a"], yssm=rng_after["max_yssm"]),
    ceilings=dict(act=E.CEIL_ACT, ssm=E.CEIL_SSM),
    acc_fp32_subset=acc_f, acc_fixed_subset=acc_fx, acc_fixed_ci95=se95,
    agreement=agree, logit_snr_db=round(snr, 2),
    encoder=dict(F_WENC=enc["f_enc"], SH_OUT=enc["sh_out"]),
    decoder=dict(F_CIN=dec["f_cin"], F_W=dec["f_w"], RQ_SHIFT=dec["rq_shift"]),
    affine=[dict(idx=m, F_BNW=q["f_bnw"], F_B=q["f_b"], SH=q["sh"],
                 W_W=B.W_BNW, W_B=B.W_BNB, max_w=q["max_w"], max_b=q["max_b"],
                 rel_worst=q["rel_worst"], acc_bound_log2=round(math.log2(max(q["acc_bound"], 1)), 1),
                 sat=(norms_fx[m].sat if m < N_LAYERS else nf_fx.sat),
                 warn=q["warn"]) for m, q in enumerate(QA_ALL)],
    layers=[dict(jury_before=q["jury_before"], jury_after=q["jury_after"], rho=q["rho"],
                 f_d=q["f_d"], bsh_min=int(q["bsh"].min()), bsh_max=int(q["bsh"].max()),
                 sat_x=sat_layers[m]["sat_x"], sat_a=sat_layers[m]["sat_a"],
                 sat_y=sat_layers[m]["sat_y"],
                 acc_bound_log2=round(math.log2(max(sat_layers[m]["acc_bound"], 1)), 1),
                 warn=q["warn"]) for m, q in enumerate(QZ)],
)
(MEM_OUT / "quant_report.json").write_text(json.dumps(report, indent=2))
 
stamp = time.strftime("%Y%m%d_%H%M")
bundle = WORK / "bundle" / f"s4d_weights_bn_{stamp}"
bundle.mkdir(parents=True, exist_ok=True)
for f in sorted(MEM_OUT.glob("*")):
    shutil.copy2(f, bundle / f.name)
for f in [BEST_CKPT, CKPT_DIR / "weights_bn.npz", RUN_LOG]:
    if Path(f).is_file():
        shutil.copy2(f, bundle / Path(f).name)
readme_txt = (
    f"Pesi S4D quantizzati per l'acceleratore VHDL -- variante BatchNorm\n"
    f"generato {time.strftime('%Y-%m-%d %H:%M')}   seed {SEED}\n"
    f"accuracy FP32 {ACC_FP32:.4f} | fixed su {len(idx)} img {acc_fx:.4f} +-{se95:.4f} (95%)\n"
    f"SNR logit {snr:.1f} dB   saturazioni totali {tot_sat}\n"
    f"riscalatura del ramo del valore: s = {s} (compensata su w della BN)\n"
    f"range calibrati su validation ({N_CAL} img)\n"
    f"weight decay escluso da: {list(SSM_KEYS)}, D, bias, BatchNorm\n\n"
    f"La norm e' un AFFINE per canale (l<m>_bn_w/l<m>_bn_b), non un LayerNorm:\n"
    f"questi .mem non sono compatibili con s4d_layernorm.\n\n{gen_str}\n"
)
(bundle / "README.txt").write_text(readme_txt)
zp = shutil.make_archive(str(bundle), "zip", root_dir=bundle)
print("archivio:", zp, f"({os.path.getsize(zp)/1e3:.0f} kB)")
 