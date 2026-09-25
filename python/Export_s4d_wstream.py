#!/usr/bin/env python3
"""Export_s4d_wstream.py -- ricompatta i .mem per-layer gia' generati
(mem_trained_sc/l<m>_*.mem) in UN solo stream a AXI_DW=64 bit, nell'ordine e
nel formato esatto che s4d_wload_axis.v si aspetta in ingresso su s_axis_w.

Non genera pesi: legge quelli GIA' scritti da Quantize_s4d_speech_command.py
(WRITE_MEM=True) e li reimpacchetta. Se quei file cambiano, va rilanciato.

Formato per layer (letto da rtl/s4d_wload_axis.v, righe 63-77), NBEAT beat:

  pos                    contenuto                    sorgente
  ---------------------  ---------------------------  ------------------------
  [0]                    bn_sh (6 bit)                 quant_report.json
                                                        affine[m].SH  (m=0..5;
                                                        l'indice 6 e' la norma
                                                        finale, FUORI da questo
                                                        stream)
  [1 .. 1+2*NCOEF)        coef, 2 beat/entry:           l<m>_s4d_coef.mem
                          beat pari  = valore[63:0]     (stesso ordine:
                          beat dispari= valore[79:64]   unita', slot, modo)
                          nei bit bassi
  [.. +H)                 {bsh[20:16], D[15:0]}         l<m>_s4d_D.mem
                          (valore GIA' impacchettato,   (contiene gia' bsh+D,
                          21 bit, nei bit bassi)         vedi write_layer_mem)
  [.. +H)                 bn_w (16 bit)                 l<m>_bn_w.mem
  [.. +H)                 bn_b (32 bit)                 l<m>_bn_b.mem
  [.. +TH)                bout (32 bit)                 l<m>_s4d_bout.mem
  [.. +NWO*WOBEAT)        wout, 12 beat/riga da 64 bit   l<m>_s4d_wout.mem
                          ciascuno, LSB-first:           (riga da 768 bit
                          beat k = riga[64k +: 64]       spezzata in 12 tocchi,
                          (vedi s4d_mixing_glu.v,         vedi s4d_mixing_glu.v
                          righe 216-224: mem[fi] =        riga 223)
                          w_file[fi][wb*LD_W +: LD_W])

Il file finale concatena i 6 layer in ordine (l0, l1, ..., l5): e' l'ordine in
cui la FSM di s4d_core_axis li richiede (wl_start si ripete per m=0..5).

Verifica: rilegge lo stream generato, lo decodifica con la STESSA logica di
s4d_wload_axis (in Python) e confronta il risultato con i valori originali
letti dai .mem per-layer. Non e' un lusso: un errore di impacchettamento qui
produce un core che carica pesi sbagliati senza nessun segnale d'errore.
"""
import json
import sys
from pathlib import Path

MEM_DIR = Path("mem_trained_sc")
OUT_FILE = MEM_DIR / "w_stream.mem"
REPORT = MEM_DIR / "quant_report.json"

H, NUNITS, NCH, MODES = 128, 8, 16, 32
LOG2_NB = 9                       # log2(NCH*MODES) = log2(512)
NCOEF = NUNITS * (1 << LOG2_NB)   # 4096
NMAC, AXI_DW = 64, 64
W_W = 12
WOBEAT = (NMAC * W_W) // AXI_DW   # 12
TH = 2 * H                        # 256
NPH = 4
NWO = NPH * H                     # 512
W_D, W_BSH = 16, 5
NLAYER = 6

H_END = 1
C_END = H_END + 2 * NCOEF
D_END = C_END + H
NW_END = D_END + H
NB_END = NW_END + H
BO_END = NB_END + TH
WO_END = BO_END + NWO * WOBEAT
NBEAT = WO_END


def read_hex_rows(path):
    return [int(t, 16) for t in Path(path).read_text().split()]


def pack_layer(m, bn_sh):
    """Restituisce la lista di NBEAT interi a 64 bit per il layer m."""
    beats = [bn_sh & 0x3F]

    coef = read_hex_rows(MEM_DIR / f"l{m}_s4d_coef.mem")
    assert len(coef) == NCOEF, f"l{m}_s4d_coef.mem: {len(coef)} righe, attese {NCOEF}"
    for v in coef:
        beats.append(v & ((1 << 64) - 1))
        beats.append((v >> 64) & 0xFFFF)

    D = read_hex_rows(MEM_DIR / f"l{m}_s4d_D.mem")
    assert len(D) == H, f"l{m}_s4d_D.mem: {len(D)} righe, attese {H}"
    beats.extend(v & ((1 << (W_D + W_BSH)) - 1) for v in D)

    bn_w = read_hex_rows(MEM_DIR / f"l{m}_bn_w.mem")
    assert len(bn_w) == H
    beats.extend(v & 0xFFFF for v in bn_w)

    bn_b = read_hex_rows(MEM_DIR / f"l{m}_bn_b.mem")
    assert len(bn_b) == H
    beats.extend(v & 0xFFFFFFFF for v in bn_b)

    bout = read_hex_rows(MEM_DIR / f"l{m}_s4d_bout.mem")
    assert len(bout) == TH, f"l{m}_s4d_bout.mem: {len(bout)} righe, attese {TH}"
    beats.extend(v & 0xFFFFFFFF for v in bout)

    wout = read_hex_rows(MEM_DIR / f"l{m}_s4d_wout.mem")
    assert len(wout) == NWO, f"l{m}_s4d_wout.mem: {len(wout)} righe, attese {NWO}"
    for row in wout:
        for k in range(WOBEAT):
            beats.append((row >> (64 * k)) & ((1 << 64) - 1))

    assert len(beats) == NBEAT, f"layer {m}: {len(beats)} beat, attesi {NBEAT}"
    return beats


def decode_layer(beats):
    """Specchio Python di s4d_wload_axis.v: dai beat ricostruisce i campi
    originali, per la verifica. Stessa aritmetica di indirizzamento dell'RTL
    (righe 89-94, 140-177 del sorgente)."""
    assert len(beats) == NBEAT
    bn_sh = beats[0]

    coef = []
    for i in range(NCOEF):
        lo, hi = beats[H_END + 2 * i], beats[H_END + 2 * i + 1]
        coef.append((hi << 64) | lo)

    D = beats[C_END:D_END]
    bn_w = beats[D_END:NW_END]
    bn_b = beats[NW_END:NB_END]
    bout = beats[NB_END:BO_END]

    wo_beats = beats[BO_END:WO_END]
    wout = []
    for r in range(NWO):
        row = 0
        for k in range(WOBEAT):
            row |= wo_beats[r * WOBEAT + k] << (64 * k)
        wout.append(row)

    return dict(bn_sh=bn_sh, coef=coef, D=D, bn_w=bn_w, bn_b=bn_b, bout=bout, wout=wout)


def main():
    if not REPORT.is_file():
        sys.exit(f"{REPORT} non trovato: serve per bn_sh per layer "
                 f"(rilanciare Quantize_s4d_speech_command.py con WRITE_MEM=True)")
    rep = json.loads(REPORT.read_text())
    sh_per_layer = [a["SH"] for a in rep["affine"][:NLAYER]]
    print(f"bn_sh per layer (da quant_report.json): {sh_per_layer}")

    all_beats = []
    for m in range(NLAYER):
        layer_beats = pack_layer(m, sh_per_layer[m])

        # --- verifica: decodifica e confronta con i .mem originali -------
        dec = decode_layer(layer_beats)
        coef0 = read_hex_rows(MEM_DIR / f"l{m}_s4d_coef.mem")
        D0 = read_hex_rows(MEM_DIR / f"l{m}_s4d_D.mem")
        bn_w0 = read_hex_rows(MEM_DIR / f"l{m}_bn_w.mem")
        bn_b0 = read_hex_rows(MEM_DIR / f"l{m}_bn_b.mem")
        bout0 = read_hex_rows(MEM_DIR / f"l{m}_s4d_bout.mem")
        wout0 = read_hex_rows(MEM_DIR / f"l{m}_s4d_wout.mem")
        assert dec["bn_sh"] == (sh_per_layer[m] & 0x3F)
        assert dec["coef"] == coef0, f"layer {m}: coef non torna"
        assert dec["D"] == [v & ((1 << 21) - 1) for v in D0], f"layer {m}: D non torna"
        assert dec["bn_w"] == [v & 0xFFFF for v in bn_w0], f"layer {m}: bn_w non torna"
        assert dec["bn_b"] == [v & 0xFFFFFFFF for v in bn_b0], f"layer {m}: bn_b non torna"
        assert dec["bout"] == [v & 0xFFFFFFFF for v in bout0], f"layer {m}: bout non torna"
        assert dec["wout"] == wout0, f"layer {m}: wout non torna"
        print(f"  layer {m}: {NBEAT} beat, verifica OK "
              f"(coef/D/bn_w/bn_b/bout/wout identici ai .mem sorgente)")

        all_beats.extend(layer_beats)

    total = NLAYER * NBEAT
    assert len(all_beats) == total
    OUT_FILE.write_text("\n".join(f"{b:016x}" for b in all_beats) + "\n")
    print(f"\nscritto {OUT_FILE}: {total} righe da 64 bit "
          f"({total * 8 / 1024:.1f} KiB), NBEAT={NBEAT} per layer x {NLAYER} layer")
    print(f"confini beat per layer: bn_sh=1 coef={2*NCOEF} D={H} bn_w={H} bn_b={H} "
          f"bout={TH} wout={NWO*WOBEAT}  (NBEAT={NBEAT})")


if __name__ == "__main__":
    main()
