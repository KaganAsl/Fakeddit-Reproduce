import subprocess
import sys
import time

# 3 gorev: her biri icin once egitim sonra degerlendirme.
# 4GB VRAM'de paralel olamaz; sirayla calisir.
TASKS = [
    ('2_way_label', 2, 'attn_2way'),
    ('3_way_label', 3, 'attn_3way'),
    ('6_way_label', 6, 'attn_6way'),
]

PY = sys.executable
NUM_WORKERS = '2'


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = (time.time() - t0) / 60
    print(f"<<< exit={rc} | {dt:.1f} dk", flush=True)
    return rc


def main():
    for label_col, num_labels, prefix in TASKS:
        print("\n" + "#" * 60, flush=True)
        print(f"# GOREV: {label_col} (num_labels={num_labels}) prefix={prefix}", flush=True)
        print("#" * 60, flush=True)

        train_cmd = [PY, 'train_multimodal_with_attention.py',
                     '--label-column', label_col, '--num-labels', str(num_labels),
                     '--output-prefix', prefix, '--num-workers', NUM_WORKERS]
        if run(train_cmd) != 0:
            print(f"!!! {prefix} EGITIM HATASI - bu gorev atlaniyor", flush=True)
            continue

        eval_cmd = [PY, 'eval_attention_model.py',
                    '--label-column', label_col, '--num-labels', str(num_labels),
                    '--output-prefix', prefix, '--epoch', '3', '--num-workers', NUM_WORKERS]
        run(eval_cmd)
        print(f"=== {prefix} TAMAM. Kaydedilen: {prefix}_epoch_1.pt, {prefix}_epoch_2.pt, {prefix}_epoch_3.pt", flush=True)

    print("\nTUM GOREVLER BITTI.", flush=True)


if __name__ == "__main__":
    main()
