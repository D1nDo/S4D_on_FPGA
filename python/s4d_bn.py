#!/usr/bin/env python3
"""
s4d_bn.py -- variante BatchNorm del modello S4D: addestramento, folding,
quantizzazione ed export dei .mem.

Perche' esiste un modulo a parte
--------------------------------
Cambiare la normalizzazione tocca UN solo punto della catena. Biquad, D,
mixing, GLU, encoder, classificatore, formati fixed point: tutto identico. Il
motore resta `s4d_export.py`; qui c'e' solo cio' che la BatchNorm cambia
davvero, piu' le tre o quattro conseguenze non ovvie.

Cosa cambia rispetto a LayerNorm
--------------------------------
1. **A inferenza la BatchNorm non e' una normalizzazione, e' un affine.**
   Con le statistiche congelate (`model.eval()`)

       u = gamma * (x - mu) / sqrt(var + eps) + beta
         = w * x + b,      w = gamma / sqrt(var + eps),  b = beta - w * mu

   cioe' una moltiplicazione e una somma per canale, con costanti note a tempo
   di export. In hardware sparisce tutto `s4d_layernorm`: niente accumulatori
   di somma e somma dei quadrati, niente `s4d_isqrt`, niente LUT 1/sqrt, niente
   latenza di due passate sul vettore. Resta un DSP e un sommatore per canale.
   Anche `EPS_NUM` sparisce dai generic: non c'e' piu' nessuna varianza a
   runtime da regolarizzare.

   Attenzione pero' a non leggere l'analogia troppo alla lettera: `w` sta al
   posto di `gamma` e ci sta in 16 bit, ma `b` NON e' `beta` e non sta in
   Q5.11. Vedi il commento su `W_BNB` sotto: e' l'errore che ha fatto scendere
   l'affine del layer 0 a 28 dB.

2. **La BatchNorm NON e' invariante di scala.** La riscalatura del ramo del
   valore -- il trucco che fa entrare le attivazioni in Q5.11 -- con LayerNorm
   era gratis perche' `LN(s*x) = LN(x)`. Qui non lo e', ma la compensazione e'
   altrettanto gratuita e per giunta *esatta*: se il residuo diventa `s*x`
   basta esportare `w/s` al posto di `w`, e `b` resta com'e'. Con LayerNorm il
   confronto prima/dopo aveva un residuo di 2.6e-3 dovuto all'eps, cioe' 1e-4 in
   relativo; qui l'unica differenza e' il riordino delle operazioni in float32
   (`(x-mu)/sqrt(var+eps)*gamma+beta` contro `w*x+b`, propagato da sei FFT su
   2048 punti): qualche ulp, ~1e-7 in relativo. Esatta in aritmetica esatta,
   non bit-identica in float32 -- e i controlli vanno scritti in ulp, non in
   relativo, o falliscono sul rumore.

3. **Il modello e' esatto solo in eval().** In addestramento la BatchNorm usa
   le statistiche del batch; i pesi che esportiamo sono quelli delle medie
   mobili. Ogni confronto float-contro-fixed di questo modulo presuppone
   `model.eval()`. Con batch 64 x 1024 passi ci sono 65536 campioni per canale
   per batch, quindi le medie mobili sono stabili -- ma la differenza esiste e
   non va confusa con un errore di quantizzazione.

4. **I .mem sono altri file.** `l<m>_bn_w.mem` / `l<m>_bn_b.mem` invece di
   `l<m>_ln_gamma.mem` / `l<m>_ln_beta.mem`. Nomi diversi apposta: se
   riusassimo i vecchi nomi, un RTL ancora con `s4d_layernorm` leggerebbe
   numeri plausibili, normalizzerebbe comunque a runtime e darebbe un risultato
   sbagliato senza alzare nessun flag. Vanno in una cartella separata
   (`mem_trained_bn/`) per lo stesso motivo.

Dipendenze: numpy, gen_s4d_mem, s4d_export (obbligatori); torch solo per il
modello e per il folding dei parametri.
"""

import math
import os

import numpy as np

import gen_s4d_mem as G
import s4d_export as E

# Il controllo va fatto QUI, all'import, non dove serve. Su Colab i moduli
# stanno su Drive e il notebook li carica solo se mancano: aggiornare s4d_bn.py
# senza aggiornare s4d_export.py e' il caso normale, non quello patologico.
# Senza questo controllo l'incompatibilita' salta fuori a meta' della cella di
# riscalatura -- cioe' dopo l'addestramento -- con un TypeError su un keyword.
API_MIN = 2
if getattr(E, "API", 1) < API_MIN:
    raise ImportError(
        f"s4d_export.py e' alla versione API {getattr(E, 'API', 1)}, serve >= {API_MIN}.\n"
        f"  file: {getattr(E, '__file__', '?')}\n"
        "La copia su Drive e' vecchia: ricaricala insieme a s4d_bn.py.\n"
        "Nel notebook basta REFRESH_MODULES = True nella prima cella, oppure:\n"
        "    from google.colab import files; import importlib\n"
        "    for n, d in files.upload().items(): (SCRIPTS / n).write_bytes(d)\n"
        "    importlib.reload(E); importlib.reload(B)"
    )

H = G.H

# ----------------------------------------------------------------------------
# Formato dell'affine
# ----------------------------------------------------------------------------
# w sta in 16 bit come gamma della LayerNorm: stesso silicio, nessun bit in piu'.
W_BNW = G.W_PAR

# b NO. La prima versione lo teneva a 16 bit nel formato dell'uscita (Q5.11,
# tetto +-16) e sommato DOPO lo shift, per analogia con beta della LayerNorm.
# E' sbagliato, e sbagliato in modo asimmetrico: b non e' beta.
#
#     b = beta - w*mu = beta - gamma * mu/sigma
#
# e mu/sigma non e' limitato da niente. Il caso peggiore e' il layer 0, il cui
# ingresso e' l'uscita dell'encoder: un solo canale d'ingresso, quindi
# x_h = w_h*p + b_h ha media b_h e deviazione |w_h|, e dove |w_h| e' piccolo il
# rapporto mu/sigma diventa enorme. Misurato su un modello addestrato:
# max|b| = 36.6 contro un tetto di 16, con l'affine del layer 0 sceso a 28 dB.
#
# Il punto e' che u = w*x + b RESTA nel range -- b grande e' cancellato da w*mu
# dentro w*x -- quindi il datapath e' sano e a non entrarci e' solo la costante.
# La correzione e' quindi di formato, non di modello: b viaggia a
# f_bnw + F_ACT bit frazionari in 32 bit e si somma PRIMA dello shift, nello
# stesso accumulatore del prodotto. E' esattamente quello che gia' si fa con
# bout nel mixing (quantizzato a FACC e sommato prima della requantizzazione).
# Costa una ROM da 32 bit invece che da 16 -- 4 kbit per norm -- e in cambio il
# bias e' anche piu' preciso di prima, perche' non passa dall'arrotondamento
# all'uscita.
W_BNB = 32

# larghezza dell'accumulatore w*x + b prima dello shift. |w|,|x| < 2^15 danno un
# prodotto sotto 2^30; sommato a b serve un bit di guardia.
W_BNACC = 34

# tetto allo shift di riallineamento. Un canale con w minuscolo (gamma ~ 0, o
# varianza enorme) e' un canale morto: portare F_BNW a 30 non lo resusciterebbe
# e chiederebbe un barrel shifter piu' largo del necessario.
SH_MAX = 24

BN_EPS = 1e-5


# ============================================================================
# Modello PyTorch
# ============================================================================
def batchnorm_seq(d_model):
    """BatchNorm1d sul canale D di un tensore (B, L, D).

    Le statistiche sono sul batch E sul tempo: 64 x 1024 campioni per canale.
    `normalized_shape` non serve a torch, serve a noi: e' l'attributo con cui
    `s4d_export.calibrate_ranges` scopre H senza sapere che norm sta guardando,
    e permette di riusare la calibrazione invece di copiarla.
    """
    import torch.nn as nn

    class BatchNormSeq(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.bn = nn.BatchNorm1d(d, eps=BN_EPS)
            self.normalized_shape = (d,)

        def forward(self, x):
            return self.bn(x.transpose(1, 2)).transpose(1, 2)

    return BatchNormSeq(d_model)


def torch_model(d_model=128, n_layers=6, d_state=64, dropout=0.2, n_classes=10):
    """Lo stesso classificatore di s4d_export.torch_model, con le norm sostituite.

    Non e' una copia del modello: e' *quel* modello con `norms` e `norm_f`
    riassegnate. Il nucleo S4D -- kernel, ZOH, GLU, l'ordine delle operazioni --
    resta letteralmente l'oggetto che la quantizzazione sa esportare. Se lo
    riscrivessi qui, la prima modifica a uno dei due lo farebbe divergere
    dall'altro senza che niente se ne accorga.
    """
    import torch.nn as nn

    m = E.torch_model(d_model, n_layers, d_state, dropout, n_classes)
    m.norms = nn.ModuleList([batchnorm_seq(d_model) for _ in range(n_layers)])
    m.norm_f = batchnorm_seq(d_model)
    return m


# ============================================================================
# Folding BatchNorm -> affine
# ============================================================================
def fold(gamma, beta, mean, var, eps=BN_EPS):
    """(gamma, beta, running_mean, running_var) -> (w, b) con u = w*x + b."""
    w = np.asarray(gamma, dtype=np.float64) / np.sqrt(np.asarray(var, dtype=np.float64) + eps)
    b = np.asarray(beta, dtype=np.float64) - w * np.asarray(mean, dtype=np.float64)
    return w, b


def weights_to_numpy(model, n_layers=6):
    """Come s4d_export.weights_to_numpy, ma le norm arrivano gia' ripiegate.

    Le chiavi delle norm sono `bn_w`/`bn_b` e `nf_w`/`nf_b`, non `gamma`/`beta`:
    un nome diverso perche' e' una cosa diversa, e perche' cosi'
    `E.quantize_layer` (che tratta gamma/beta come opzionali) non le prende per
    sbaglio e non le quantizza nel formato della LayerNorm.
    """
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    nf_w, nf_b = fold(sd["norm_f.bn.weight"], sd["norm_f.bn.bias"],
                      sd["norm_f.bn.running_mean"], sd["norm_f.bn.running_var"])
    W = dict(enc_w=sd["input_proj.weight"].reshape(-1),
             enc_b=sd["input_proj.bias"].reshape(-1),
             nf_w=nf_w, nf_b=nf_b,
             cls_w=sd["classifier.weight"], cls_b=sd["classifier.bias"], layers=[])
    for m in range(n_layers):
        p = f"layers.{m}."
        c = sd[p + "kernel.C"]
        bw, bb = fold(sd[f"norms.{m}.bn.weight"], sd[f"norms.{m}.bn.bias"],
                      sd[f"norms.{m}.bn.running_mean"], sd[f"norms.{m}.bn.running_var"])
        W["layers"].append(dict(
            log_A_real=sd[p + "kernel.log_A_real"], A_imag=sd[p + "kernel.A_imag"],
            log_dt=sd[p + "kernel.log_dt"], C=c[..., 0] + 1j * c[..., 1],
            D=sd[p + "D"], wout=sd[p + "output_linear.0.weight"][:, :, 0],
            bout=sd[p + "output_linear.0.bias"],
            bn_w=bw, bn_b=bb))
    return W


def numpy_to_torch_(model, W):
    """Riscrive i pesi torch dal dizionario numpy. Valido SOLO in eval().

    L'affine ripiegato non ha un'inversa unica in (gamma, beta, mu, var): ne
    scegliamo la piu' semplice, mu=0 e var=1-eps, cosi' sqrt(var+eps)=1 e il
    modulo BatchNorm in eval calcola esattamente w*x + b. In train() ricomincerebbe
    a usare le statistiche del batch e questa identita' salterebbe: il modello
    ricostruito serve per verificare la riscalatura, non per riaddestrare.
    """
    import torch
    with torch.no_grad():
        E.numpy_to_torch_(model, W, with_norm=False)   # tutto cio' che non e' norm
        sd = model.state_dict()
        sd["norm_f.bn.weight"].copy_(torch.tensor(W["nf_w"]))
        sd["norm_f.bn.bias"].copy_(torch.tensor(W["nf_b"]))
        sd["norm_f.bn.running_mean"].zero_()
        sd["norm_f.bn.running_var"].fill_(1.0 - BN_EPS)
        for m, d in enumerate(W["layers"]):
            sd[f"norms.{m}.bn.weight"].copy_(torch.tensor(d["bn_w"]))
            sd[f"norms.{m}.bn.bias"].copy_(torch.tensor(d["bn_b"]))
            sd[f"norms.{m}.bn.running_mean"].zero_()
            sd[f"norms.{m}.bn.running_var"].fill_(1.0 - BN_EPS)
    return model


# ============================================================================
# Riscalatura del ramo del valore -- esatta, qui
# ============================================================================
def rescale_value_branch(W, s):
    """Scala di s il residuo senza cambiare la funzione calcolata.

    Con LayerNorm bastava scalare encoder e ramo "a" del mixing, perche' la norm
    assorbiva la scala da sola. La BatchNorm no: se il suo ingresso diventa s*x,
    l'uscita w*(s*x) + b non e' piu' quella di prima. La compensazione e' una
    riga -- esportare w/s -- e a differenza del caso LayerNorm e' esatta: non
    c'e' nessun eps che rompa l'omogeneita'. In float32 resta comunque il
    riordino delle operazioni, qualche ulp sui logit: chi verifica lo faccia
    con una soglia in ulp, non in relativo, o fallisce sul rumore.

    s dovrebbe restare una potenza di due, cosi' w/s sposta solo l'esponente e
    non aggiunge errore di arrotondamento ai pesi.
    """
    out = E.rescale_value_branch(W, s)
    out["nf_w"] = np.asarray(W["nf_w"], dtype=np.float64) / s
    for e, d in zip(out["layers"], W["layers"]):
        e["bn_w"] = np.asarray(d["bn_w"], dtype=np.float64) / s
        e["bn_b"] = np.asarray(d["bn_b"], dtype=np.float64).copy()
    return out


# ============================================================================
# Quantizzazione dell'affine
# ============================================================================
def quantize_affine(w, b, f_out=None, name="bn", verbose=True):
    """w, b float (H,) -> interi + lo shift di riallineamento.

    Il datapath e':

        u = sat( round( (w_q * x_q + b_q) / 2^SH ) ),   SH = F_BNW + F_ACT - F_OUT

    con `b` quantizzato a F_BNW + F_ACT bit frazionari, cioe' nel formato
    dell'accumulatore, e sommato PRIMA dello shift. Vedi il commento su W_BNB in
    testa al file: sommarlo dopo, a 16 bit, lo taglia a +-16 e quel tetto b non
    lo rispetta -- non perche' il modello esca dal datapath, ma perche' b e'
    grande e opposto a w*mu, e i due si cancellano solo DOPO essere stati
    sommati.

    F_BNW e' scelto per riempire i 16 bit di w, e abbassato se b non entra nei
    32 bit dell'accumulatore. Abbassarlo costa precisione su w, ma e' l'unico
    dei due che si puo' pagare: un b tosato non e' un arrotondamento, e' un
    canale che calcola un'altra funzione.
    """
    f_out = G.F_U if f_out is None else f_out
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    max_b = float(np.abs(b).max())

    f_bnw = G.calib(w, W_BNW)
    lim_b = (1 << (W_BNB - 1)) - 1
    while f_bnw > 0 and max_b * 2.0 ** (f_bnw + G.F_ACT) > lim_b:
        f_bnw -= 1
    sh = f_bnw + G.F_ACT - f_out
    if sh < 0:
        raise ValueError(f"{name}: SH negativo ({sh}). I pesi della norm sono "
                         f"troppo grandi per {W_BNW} bit: max|w| = {np.abs(w).max():.3f}")
    if sh > SH_MAX:
        f_bnw -= sh - SH_MAX
        sh = SH_MAX
    f_b = f_bnw + G.F_ACT
    wq = G.q(w, f_bnw, W_BNW)
    bq = G.q(b, f_b, W_BNB)

    warn = []
    if max_b * 2.0 ** f_b > lim_b:
        n = int((np.abs(b) * 2.0 ** f_b > lim_b).sum())
        warn.append(f"{name}: {n}/{b.size} bias tosati anche a {W_BNB} bit "
                    f"(max|b| = {max_b:.4f}): allargare W_BNB, non clippare")
    # bound sull'accumulatore: se sfora, il sommatore dell'RTL va allargato
    acc = int(np.abs(wq).max()) * ((1 << (G.W_ACT - 1)) - 1) + int(np.abs(bq).max())
    if acc >= (1 << (W_BNACC - 1)):
        warn.append(f"{name}: accumulatore w*x+b fino a 2^{math.log2(acc):.1f}, "
                    f"non entra in {W_BNACC} bit")
    # errore relativo del quanto sul canale piu' debole: e' la spia di un F_BNW
    # dominato da un solo canale fuori scala
    nz = np.abs(w) > 0
    rel = (0.5 / 2.0 ** f_bnw) / np.abs(w[nz]).min() if nz.any() else 0.0

    qa = dict(w=wq, b=bq, f_bnw=f_bnw, f_b=f_b, f_out=f_out, sh=sh, warn=warn,
              max_w=float(np.abs(w).max()), max_b=max_b,
              acc_bound=acc, rel_worst=float(rel))
    if verbose:
        print(f"  {name}: F_BNW {f_bnw}  F_B {f_b}  SH {sh}   max|w| {np.abs(w).max():.4f}"
              f"  max|b| {max_b:.4f}   quanto/|w|min {rel:.1%}"
              f"   acc 2^{math.log2(max(acc,1)):.1f}/{W_BNACC-1}")
        for x in warn:
            print("  ATTENZIONE  " + x)
    return qa


def write_affine_mem(m, qa, outdir, verbose=True):
    """l<m>_bn_w.mem (16 bit), l<m>_bn_b.mem (32 bit).

    Lo shift non sta nei .mem: e' un generic. Le due ROM hanno larghezze
    diverse, e non e' un dettaglio da dedurre dal file -- un lettore che
    assumesse 16 bit anche per b leggerebbe due parole per canale e sbaglierebbe
    tutto senza accorgersene.
    """
    o = str(outdir)
    msgs = [E._write(os.path.join(o, f"l{m}_bn_w.mem"), [G.hx(v, W_BNW) for v in qa["w"]]),
            E._write(os.path.join(o, f"l{m}_bn_b.mem"), [G.hx(v, W_BNB) for v in qa["b"]])]
    if verbose:
        print("\n".join(msgs))


# ============================================================================
# Specchio bit-accurate dell'affine
# ============================================================================
class AffineFX:
    """u = sat( shr_round(w*x + b, SH) ). Conta le saturazioni.

    E' la funzione che si passa a `E.LayerFX(..., norm=...)` e a
    `E.run_chain_fx(..., norms=..., norm_final=...)`: il resto del datapath resta
    quello verificato contro il Verilog, e non viene ricopiato qui.

    `sat` conta i campioni in cui u esce da Q5.11. Un conteggio non nullo qui
    vuol dire una cosa diversa dalle saturazioni degli altri siti: non e' il
    formato di b (che ormai ha 32 bit), e' l'uscita della norm che esce davvero
    dal datapath.
    """

    def __init__(self, qa, w_out=None):
        self.q = qa
        self.w_out = G.W_U if w_out is None else w_out
        self.sat = 0
        self.acc_bound = 0

    def __call__(self, x):
        q = self.q
        acc = q["w"][None, :] * x + q["b"][None, :]
        self.acc_bound = max(self.acc_bound, int(np.abs(acc).max()))
        v = E.shr_r(acc, q["sh"])
        lim = (1 << (self.w_out - 1)) - 1
        self.sat += int((np.abs(v) > lim).sum())
        return E.satn(v, self.w_out)


def affine_float(x, w, b):
    """Riferimento float dell'affine, per il self-check."""
    return np.asarray(x, dtype=np.float64) * np.asarray(w) + np.asarray(b)


def collect_norm_inputs(model, loader, device, n_images=4, n_layers=6,
                        stride=8, max_rows=4096):
    """Campiona gli ingressi REALI di ogni norm. Ritorna n_layers+1 array (n, H).

    Serve perche' non esiste uno stimolo sintetico corretto per questo blocco.
    Un x uniforme o gaussiano sul range globale e' fisicamente impossibile:
    w_h = gamma_h/sigma_h e' grande esattamente dove sigma_h e' piccolo, e in
    quei canali l'attivazione vera vive in mu_h +- pochi sigma_h, non su tutto
    il range. Spazzare l'intero range manda quei canali fuori scala per
    costruzione, e il self-check finisce per misurare l'assurdita' dello
    stimolo invece della bonta' della quantizzazione -- con percentuali di
    saturazione a due cifre che il modello vero non produce mai.

    stride sottocampiona il tempo: i 1024 passi di un'immagine sono fortemente
    correlati, uno ogni 8 copre la stessa distribuzione a un ottavo della RAM.
    """
    import torch
    store = [[] for _ in range(n_layers + 1)]
    hs = []

    def pre(i):
        def f(_, inp):
            v = inp[0].detach()                      # (B, L, H)
            v = v[:, ::stride, :].reshape(-1, v.shape[-1])
            store[i].append(v.double().cpu().numpy())
        return f

    for m in range(n_layers):
        hs.append(model.norms[m].register_forward_pre_hook(pre(m)))
    hs.append(model.norm_f.register_forward_pre_hook(pre(n_layers)))

    model.eval()
    seen = 0
    try:
        with torch.no_grad():
            for x, _ in loader:
                if seen >= n_images:
                    break
                x = x[: n_images - seen].to(device)
                model(x)
                seen += x.shape[0]
    finally:
        for h in hs:
            h.remove()
    return [np.concatenate(s)[:max_rows] for s in store]


def diagnose_affine(xs, w, b, qa, top=6):
    """Perche' un affine perde SNR: il canale, e quale dei due limiti lo lega.

    Ci sono due termini di errore su u = w*x + b, e hanno rimedi opposti:

      A) risoluzione dell'INGRESSO. x arriva gia' quantizzato in Q5.11, con
         quanto q = 2^-F_ACT. L'errore w*q/sqrt(12) va confrontato con
         l'ampiezza utile |w|*sigma_h, e il rapporto si semplifica:

             SNR_max(h) ~ sigma_h / q * sqrt(12)

         cioe' dipende SOLO da quanti quanti di Q5.11 occupa la deviazione di
         quel canale. Un canale con sigma sotto il quanto e' gia' morto nel
         residuo: nessun formato dell'affine puo' recuperarlo, e allargare
         F_BNW non serve a niente.

      B) quanto di W. L'errore dw*|x| ~ 2^-(F_BNW+1) * |mu_h| confrontato con
         |w|*sigma_h. Questo si', dipende da F_BNW -- e cresce con |mu_h|/sigma_h,
         cioe' e' la cancellazione fra w*x e b che si mangia le cifre.

    Se domina A il problema e' la riscalatura o il formato del residuo; se
    domina B e' F_BNW, e il rimedio e' un esponente per canale come il bsh del
    biquad. Confonderli fa perdere giornate.
    """
    x = np.asarray(xs, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    sig = x.std(axis=0)
    mu = np.abs(x.mean(axis=0))
    q_x = 2.0 ** -G.F_ACT
    dw = 2.0 ** -(qa["f_bnw"] + 1)

    amp = np.abs(w) * np.maximum(sig, 1e-30)          # ampiezza utile di u
    err_a = np.abs(w) * q_x / math.sqrt(12)
    err_b = dw * mu
    snr_a = 20 * np.log10(amp / np.maximum(err_a, 1e-300))
    snr_b = 20 * np.log10(amp / np.maximum(err_b, 1e-300))
    snr = np.minimum(snr_a, snr_b)

    order = np.argsort(snr)[:top]
    print(f"  canali peggiori (quanto del residuo q = {q_x:.2e}, "
          f"quanto di w = {dw:.2e})")
    print(f"  {'h':>4s}{'sigma/q':>10s}{'|mu|/sigma':>12s}{'|w|':>10s}"
          f"{'SNR_in':>9s}{'SNR_w':>8s}{'limite':>9s}")
    for h in order:
        lim = "ingresso" if snr_a[h] <= snr_b[h] else "F_BNW"
        print(f"  {h:4d}{sig[h]/q_x:10.2f}{mu[h]/max(sig[h],1e-30):12.1f}"
              f"{abs(w[h]):10.2f}{snr_a[h]:8.1f}dB{snr_b[h]:7.1f}dB{lim:>9s}")
    dead = int((sig < q_x).sum())
    print(f"  canali con sigma sotto il quanto del residuo: {dead}/{sig.size}"
          + ("  <- morti nel formato, non nell'affine" if dead else ""))
    return dict(snr_in=snr_a, snr_w=snr_b, sigma=sig, mu=mu, dead=dead,
                limited_by=("ingresso" if snr_a[order[0]] <= snr_b[order[0]] else "F_BNW"))


# ============================================================================
# Decoder
# ============================================================================
def write_decoder_affine_mem(nf_w, nf_b, outdir, nlayer=6, verbose=True):
    """Norm finale come affine, indice nlayer per restare coerente con mem_name."""
    qa = quantize_affine(nf_w, nf_b, f_out=G.F_ACT, name=f"l{nlayer}_bn", verbose=verbose)
    write_affine_mem(nlayer, qa, outdir, verbose=verbose)
    return qa


# ============================================================================
# Rilettura dei .mem -- la catena ricostruita dai bit, non dai float
# ============================================================================
def _hex_rows(path):
    return [int(t, 16) for t in open(str(path)).read().split()]


def _signed(words, lo, w):
    m, half = (1 << w) - 1, 1 << (w - 1)
    v = np.array([(x >> lo) & m for x in words], dtype=np.int64)
    return np.where(v >= half, v - (1 << w), v)


# La permutazione delle colonne di W_out vive in un posto solo, E.wout_row_perm
# (oggi l'identita': il riordino dei canali e' passato in hardware). Qui c'era
# una seconda copia scritta a mano, pari-poi-dispari: due definizioni della
# stessa convenzione sono esattamente il modo in cui la rilettura smette di
# verificare l'export e comincia a confermarne l'errore.
WOUT_PERM = E.wout_row_perm()


def read_mem_dir(mem_dir, report, n_layers=6):
    """Ricostruisce (QZ, QA, enc, dec, luts) leggendo i .mem come li legge l'RTL.

    Perche' dai .mem e non dal .npz: riquantizzare i pesi float verifica la
    matematica ma NON il packing. Se `s4d_coef.mem` mettesse a2 dove il VHDL
    legge a1, o se `s4d_wout.mem` associasse la riga (r*H + h) al canale
    sbagliato, una riquantizzazione non se ne accorgerebbe. Qui i campi si
    estraggono con gli stessi indici di bit del VHDL.

    `report` e' quant_report.json: serve per i parametri che NON stanno nei
    .mem -- F_SSM, f_d per layer, gli shift. Il fatto che f_d sia per layer e
    l'RTL lo prenda come generic globale e' un problema aperto, vedi
    check_global_f_d().
    """
    mem = str(mem_dir).rstrip("/") + "/"
    E.set_f_ssm(int(report["f_ssm"]))

    QZ = []
    for m in range(n_layers):
        coef = _hex_rows(mem + f"l{m}_s4d_coef.mem")
        assert len(coef) == H * MODES_, f"l{m}_s4d_coef.mem: {len(coef)} righe"
        r = lambda v: v.reshape(H, MODES_)
        a1 = r(_signed(coef, 0, G.W_A))
        a2 = r(_signed(coef, G.W_A, G.W_A))
        b0 = r(_signed(coef, 2 * G.W_A, G.W_B))
        b1 = r(_signed(coef, 2 * G.W_A + G.W_B, G.W_B))

        dw = _hex_rows(mem + f"l{m}_s4d_D.mem")
        D = _signed(dw, 0, G.W_D)
        bsh = np.array([(x >> G.W_D) & ((1 << E.W_BSH) - 1) for x in dw], dtype=np.int64)

        # s4d_wout.mem: NPH*H righe da NMAC*W_W bit, la riga (r*H + h) porta i
        # pesi degli NMAC canali di uscita della fase r per il canale di
        # ingresso h. Il parsing sta in E.read_wout_mem, che e' anche quello
        # che l'export usa per rileggersi: una definizione sola dei bit.
        wout = E.read_wout_mem(mem + f"l{m}_s4d_wout.mem")
        bout = _signed(_hex_rows(mem + f"l{m}_s4d_bout.mem"), 0, G.W_MACC)

        f_d = int(report["layers"][m]["f_d"])
        QZ.append(dict(a1=a1, a2=a2, b0=b0, b1=b1, bsh=bsh, D=D, f_d=f_d,
                       sh_d=f_d + G.F_U - G.F_Y, wout=wout, bout=bout))

    QA = []
    for m in range(n_layers + 1):
        a = report["affine"][m]
        QA.append(dict(w=_signed(_hex_rows(mem + f"l{m}_bn_w.mem"), 0, W_BNW),
                       b=_signed(_hex_rows(mem + f"l{m}_bn_b.mem"), 0, W_BNB),
                       f_bnw=int(a["F_BNW"]), f_b=int(a["F_B"]),
                       sh=int(a["SH"]),
                       f_out=G.F_ACT if m == n_layers else G.F_U))

    enc = dict(w=_signed(_hex_rows(mem + "enc_w.mem"), 0, E.W_WENC),
               b=_signed(_hex_rows(mem + "enc_b.mem"), 0, E.W_BENC),
               f_enc=int(report["encoder"]["F_WENC"]),
               sh_out=int(report["encoder"]["SH_OUT"]))

    crows = _hex_rows(mem + "cls_w.mem")
    cw = np.zeros((E.K_CLS, H), dtype=np.int64)
    for h, word in enumerate(crows):
        col = np.array([(word >> (c * E.W_CLS)) & 0xFF for c in range(E.K_CLS)],
                       dtype=np.int64)
        cw[:, h] = np.where(col >= 128, col - 256, col)
    dec = dict(w=cw, b=_signed(_hex_rows(mem + "cls_b.mem"), 0, E.W_ACC_D),
               f_w=int(report["decoder"]["F_W"]), f_cin=int(report["decoder"]["F_CIN"]),
               rq_shift=int(report["decoder"]["RQ_SHIFT"]))

    def _lut(name):
        w = _hex_rows(mem + name)
        base = np.array([x & 0xFFFF for x in w], dtype=np.int64)
        slope = np.array([(x >> 16) & 0xFFFF for x in w], dtype=np.int64)
        f = lambda v: np.where(v >= 1 << 15, v - (1 << 16), v)
        return f(base), f(slope)

    gb, gs = _lut("gelu_lut.mem")
    sb, ss = _lut("sigmoid_lut.mem")
    luts = dict(gelu_b=gb, gelu_s=gs, sig_b=sb, sig_s=ss,
                isqrt=np.zeros(256, dtype=np.int64))   # non serve senza LayerNorm
    return QZ, QA, enc, dec, luts


MODES_ = G.MODES


def check_jury(QZ, verbose=True):
    """Poli dentro il cerchio unitario, verificato SUI BIT LETTI.

    Il clamp e' gia' stato applicato in export, ma controllarlo qui costa niente
    ed e' l'unico modo di sapere che e' arrivato fin dentro i .mem. Un solo
    biquad fuori e il layer diventa un integratore.
    """
    one = 1 << G.F_A
    tot = 0
    for m, q in enumerate(QZ):
        bad = int(((np.abs(q["a1"]) >= one + q["a2"]) | (q["a2"] >= one)).sum())
        rho = float(np.sqrt(np.maximum(q["a2"], 0) / one).max())
        tot += bad
        if verbose:
            print(f"  layer {m}: fuori dal triangolo {bad}/{q['a1'].size}"
                  f"   max|lambda_bar| {rho:.6f}")
    if verbose:
        print("  OK: tutti i poli dentro il cerchio unitario" if tot == 0
              else f"  ATTENZIONE: {tot} biquad instabili")
    return tot


def check_global_f_d(QZ, verbose=True):
    """F_D nell'RTL e' un generic GLOBALE: qui i sei layer devono concordare.

    `SH_D = F_D + F_U - F_Y` e' una costante di s4d_biquad_unit, una sola per
    tutto il design. Se l'export ha calibrato F_D per layer -- e `G.calib` lo
    fa, dando valori diversi -- allora qualunque valore si passi all'RTL e'
    sbagliato su qualche layer, e il ramo D di quei layer esce scalato per una
    potenza di due. Non lo dice nessun flag: i .mem sono in un formato e il
    datapath in un altro.

    Non e' teoria. Misurato su questo modello, con D esportato a f_d per layer
    [13,13,13,14,14,13]: passando F_D=13 l'SNR sui logit scende da 33 dB a
    7.0, passando F_D=14 a 4.9, e la classe predetta cambia su immagini vere.

    Il rimedio sta in export, non nell'RTL: quantizzare tutti i D con lo stesso
    f_d, e sceglierlo pari al MINIMO fra i calibrati. Il minimo perche' un f_d
    piu' alto del calibrato tosa D -- con f_d=14 su questo modello sarebbero
    stati tosati 7 canali -- e un canale tosato e' un modello diverso, mentre
    un bit di risoluzione in meno su due layer costa 52 dB rispetto al
    riferimento, cioe' venti dB sotto il rumore di quantizzazione: invisibile.
    """
    fds = [q["f_d"] for q in QZ]
    ok = len(set(fds)) == 1
    if verbose:
        print(f"  F_D per layer: {fds}")
        print(f"  OK: un solo valore ({fds[0]}), il generic globale e' coerente" if ok else
              "  ATTENZIONE: valori diversi. Questi .mem NON possono essere usati con\n"
              "  nessun F_D globale. Riesportare con f_d = min(f_d) su tutti i layer\n"
              f"  (qui sarebbe {min(fds)}): e' un parametro di quantize_layer.")
    return ok


# ============================================================================
# Generic
# ============================================================================
def required_generics(enc, dec, qas, f_ssm=None, nlayer=6, mem_dir="mem_trained_bn/",
                      f_d=None):
    """I generic per s4d_top nella variante BatchNorm.

    Due differenze rispetto alla versione LayerNorm, ed entrambe sono guadagni:

    * `EPS_NUM` **non serve piu'**. Regolarizzava la varianza calcolata a
      runtime; con la BatchNorm ripiegata non c'e' nessuna varianza a runtime.
      L'eps e' gia' dentro w = gamma/sqrt(var+eps), a tempo di export.
    * compaiono `BN_SH_<m>`, uno per norm (sei layer piu' il decoder). Sono
      scalari: lo shift di riallineamento dell'accumulatore w*x + b. Un valore
      sbagliato NON e' silenzioso solo se lo si passa; se si lascia il default
      il risultato e' scalato di una potenza di due, che e' esattamente il tipo
      di errore che passa inosservato in un classificatore -- l'argmax non
      cambia quasi mai, l'SNR crolla.

    Compaiono anche `NUNITS`, `NCH` e `NMAC`, che descrivono la ripiegatura del
    datapath (banco 8x16 invece di 64x2, mixing a 64 MAC invece di 256). Non
    cambiano nessun formato numerico, ma `NMAC` determina la forma di
    `s4d_wout.mem` -- NPH*H righe da NMAC*W_W bit -- quindi un valore diverso da
    quello con cui i .mem sono stati generati fa leggere righe sbagliate.

    E una trappola: `BN_W_B` e' **32**, non 16. Le due ROM della norm hanno
    larghezze diverse perche' b non e' beta e non sta in Q5.11 (vedi il commento
    su W_BNB in testa al file). Un lettore che assumesse 16 bit anche per b
    leggerebbe due parole per canale, e sbaglierebbe tutto in silenzio.
    """
    f_ssm = G.F_SSM if f_ssm is None else f_ssm
    rows = [
        ("MEM_DIR", f'"{mem_dir}"', None),
        ("NORM_KIND", '"affine"', "BatchNorm ripiegata: niente s4d_layernorm"),
        ("PER_LAYER_MEM", "true", "i sei layer hanno pesi diversi"),
        ("WITH_LOADER", "true", "false in sintesi definitiva"),
        ("NUNITS", G.NUNITS, "banco ripiegato: %d unita' (era 64)" % G.NUNITS),
        ("NCH", G.NCH, "%d canali per unita' (era 2)" % G.NCH),
        ("NMAC", G.NMAC, "mixing ripiegato: %d MAC, NPH = TH/NMAC = %d fasi"
                         % (G.NMAC, G.NPH)),
        ("F_WENC", enc["f_enc"], "default 23; SH_OUT = F_WENC-F_ACT = %d" % enc["sh_out"]),
        ("F_SSM", f_ssm, "default 11"),
        ("RQ_SHIFT", dec["rq_shift"], "default 7"),
    ]
    if f_d is not None:
        rows.append(("F_D", int(f_d),
                     "default 14: TOSA D su meta' dei layer"))
    rows.append(("BN_W_W", W_BNW, "ROM di w"))
    rows.append(("BN_W_B", W_BNB, "ROM di b: 32 bit, NON 16 -- si somma prima dello shift"))
    for m, qa in enumerate(qas):
        tag = "decoder" if m == nlayer else f"layer {m}"
        rows.append((f"BN_SH_{m}", qa["sh"], f"{tag}: F_BNW {qa['f_bnw']}, F_B {qa['f_b']}"))
    out = ["generic map ("]
    for k, v, c in rows:
        line = f"    {k:<14s}=> {v},"
        out.append(line if c is None else f"{line:<44s}-- {c}")
    out.append(")")
    out += ["",
            f"-- decoder: F_CIN {dec['f_cin']}, F_W {dec['f_w']}",
            "-- EPS_NUM non e' piu' necessario: nessuna varianza calcolata a runtime.",
            "-- s4d_layernorm va sostituito da un affine per canale (un DSP e un",
            "-- sommatore): e' l'unica modifica all'RTL che la BatchNorm richiede."]
    return "\n".join(out)


def print_affine_table(qas, nlayer=6):
    print(f"{'norm':>9s}{'F_BNW':>7s}{'F_B':>6s}{'SH':>5s}{'max|w|':>10s}{'max|b|':>10s}"
          f"{'quanto/|w|min':>15s}{'acc':>8s}")
    for m, qa in enumerate(qas):
        nm = "decoder" if m == nlayer else f"layer {m}"
        print(f"{nm:>9s}{qa['f_bnw']:7d}{qa['f_b']:6d}{qa['sh']:5d}{qa['max_w']:10.4f}"
              f"{qa['max_b']:10.4f}{qa['rel_worst']:14.2%}"
              f"   2^{math.log2(max(qa['acc_bound'], 1)):.1f}")
    print(f"  (max|b| oltre 16 e' normale e innocuo: b sta a {W_BNB} bit e si somma "
          f"prima dello shift)")
