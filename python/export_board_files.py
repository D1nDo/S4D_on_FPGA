#!/usr/bin/env python3
"""
export_board_files.py -- file del test in scheda (s4d_PTB_XL_2158ECG.c).

Si lancia da python/, con PTB-XL 1.0.3 (records100) in data_ptbxl/:

    python export_board_files.py                      # stimoli + etichette
    python export_board_files.py --mem <quant_*/mem>  # anche lo stream dei pesi

Scrive nella cartella di uscita (default ../board_test):

  samples_2158ECG.bin  fold 10, 2158 ECG con almeno una superclasse, in ordine
                       di ecg_id; 1024 passi x 12 derivazioni, int16 Q5.11
                       little-endian, ordine [n][t][c] (24576 byte per ECG)
  labels_2158ECG.bin   2158 byte, maschera di bit: bit0 CD, bit1 HYP, bit2 MI,
                       bit3 NORM, bit4 STTC (etichette multi-label)
  w_stream_ECG.bin     solo con --mem: pesi dei 6 layer, 64 bit/beat
                       little-endian, nell'ordine di s4d_wload_axis (718896 byte)

STIMOLI: NON serve il training. Si leggono i record grezzi e si applica la
stessa preelaborazione del training (Train_s4d_ecg.preprocess: 100 Hz,
mediana per derivazione, scala fissa, Q5.11, 24 zeri di coda). L'unico
parametro e' la SCALE della scala fissa, che deve essere quella del modello:
  1. --scale X, se data;
  2. altrimenti "input_scale" in <mem>/quant_report.json, se c'e';
  3. altrimenti la si ricalcola con Train_s4d_ecg.calibrate_scale sugli stessi
     record di training (fold 1-8, seed fisso): la calibrazione e'
     deterministica e da' lo stesso valore del training.

STREAM: impacchettato da Export_s4d_wstream.pack_layer e ridecodificato con
decode_layer contro i .mem sorgente prima di essere scritto: un errore di
packing produrrebbe un core che carica pesi sbagliati senza nessun segnale.
Serve la cartella completa prodotta da Quantize_s4d_ecg.py (tutti i layer);
i 12 .mem di ../mem da soli non bastano.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import Export_s4d_wstream as X
import Train_s4d_ecg as T


# ============================================================================
# stimoli ed etichette, dai record grezzi
# ============================================================================
def resolve_scale(args):
    if args.scale is not None:
        return float(args.scale), "--scale"
    if args.mem is not None:
        rep_path = Path(args.mem) / "quant_report.json"
        if rep_path.is_file():
            s = json.loads(rep_path.read_text()).get("input_scale")
            if s is not None:
                return float(s), str(rep_path)
    tab = T.superclass_table(T.DATA, label_mode="multi")
    tr = tab[tab.strat_fold.isin(T.TRAIN_FOLDS)].file.to_numpy()
    return T.calibrate_scale(tr, T.DATA), "ricalcolata sul training (fold 1-8)"


def build_stimuli(scale):
    assert T.FS == 100 and T.L == 1024, "Train_s4d_ecg.py deve essere a FS=100, L=1024"
    assert T.NORM_MODE == "fixed", "gli stimoli della scheda usano la scala fissa"
    tab = T.superclass_table(T.DATA, label_mode="multi")
    test = tab[tab.strat_fold.isin(T.TEST_FOLDS)]
    missing = [f for f in test.file if not (T.DATA / f"{f}.dat").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} record mancanti (es. {T.DATA / missing[0]}.dat): "
                         f"scaricare PTB-XL 1.0.3 records100 in {T.DATA}/")
    out = [T.preprocess(T.DATA / f"{f}.dat", "fixed", scale) for f in test.file]
    samples = np.stack([o[0] for o in out])                      # (N, L, 12) int16
    n_sat = sum(o[1] for o in out)
    bits = 1 << np.arange(len(T.CLASSES), dtype=np.uint8)
    mask = (test[T.CLASSES].to_numpy(dtype=np.uint8) * bits).sum(1).astype(np.uint8)
    print(f"stimoli: {len(test)} record del fold {T.TEST_FOLDS}, campioni saturati "
          f"in Q5.11 {n_sat} ({100 * n_sat / samples[:, :T.RAW_L].size:.4f}%)")
    print("positivi per classe: " + "  ".join(
        f"{c} {int(test[c].sum())}" for c in T.CLASSES))
    return samples.astype("<i2").tobytes(), mask.tobytes(), len(test)


# ============================================================================
# stream dei pesi, dai .mem quantizzati
# ============================================================================
def build_stream(mem_dir):
    rep = json.loads((mem_dir / "quant_report.json").read_text())
    bn_sh = [a["SH"] for a in rep["affine"][:X.NLAYER]]
    X.MEM_DIR = mem_dir
    beats = []
    for m in range(X.NLAYER):
        lb = X.pack_layer(m, bn_sh[m])
        dec = X.decode_layer(lb)
        rd = lambda n: X.read_hex_rows(mem_dir / f"l{m}_{n}.mem")
        assert dec["bn_sh"] == (bn_sh[m] & 0x3F)
        assert dec["coef"] == rd("s4d_coef"), f"layer {m}: coef"
        assert dec["D"] == [v & ((1 << 21) - 1) for v in rd("s4d_D")], f"layer {m}: D"
        assert dec["bn_w"] == [v & 0xFFFF for v in rd("bn_w")], f"layer {m}: bn_w"
        assert dec["bn_b"] == [v & 0xFFFFFFFF for v in rd("bn_b")], f"layer {m}: bn_b"
        assert dec["bout"] == [v & 0xFFFFFFFF for v in rd("s4d_bout")], f"layer {m}: bout"
        assert dec["wout"] == rd("s4d_wout"), f"layer {m}: wout"
        beats.extend(lb)
    print(f"stream: bn_sh per layer {bn_sh}, {len(beats)} beat, verifica di packing OK")
    return np.array(beats, dtype=np.uint64).astype("<u8").tobytes()


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mem", type=Path, default=None,
                    help="cartella quant_*/mem di Quantize_s4d_ecg.py: genera anche "
                         "w_stream_ECG.bin e legge input_scale da quant_report.json")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "board_test",
                    help="cartella di uscita (default ../board_test)")
    ap.add_argument("--scale", type=float, default=None,
                    help="SCALE della scala fissa in 1/mV (default: vedi docstring)")
    args = ap.parse_args()

    scale, src = resolve_scale(args)
    print(f"SCALE {scale!r} /mV  ({src})")
    args.out.mkdir(parents=True, exist_ok=True)

    samples, labels, n = build_stimuli(scale)
    (args.out / f"samples_{n}ECG.bin").write_bytes(samples)
    (args.out / f"labels_{n}ECG.bin").write_bytes(labels)
    if args.mem is not None:
        (args.out / "w_stream_ECG.bin").write_bytes(build_stream(args.mem))

    for f in sorted(args.out.glob("*.bin")):
        print(f"  {f}  {f.stat().st_size} byte")


if __name__ == "__main__":
    main()
