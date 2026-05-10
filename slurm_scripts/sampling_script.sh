#!/usr/bin/env bash
set -euo pipefail

# Minimal: download mmCIF(.gz) for ALL PDB IDs in splits/design_{train,valid,test}.json
# then run NA-MPNN preprocessing on them.
#
# Run:
#   cd /path/to/NA-MPNN
#   bash scripts/download_and_preprocess_design_splits.sh
#
# Optional knobs:
#   OUT_DIR=./data/datasets/design_from_splits
#   N_DOWNLOAD=8     # parallel downloads
#   N_PREPROC=8      # parallel preprocess workers (modulo/remainder)
#   SPLITS="train,valid,test"   # which split files to use

# Repo root is the directory containing this script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Ensure repo root and pdbx package are on PYTHONPATH so local imports resolve.
export PYTHONPATH="${REPO_ROOT}/pdbx:${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/datasets/design_from_splits}"
STRUCT_DIR="${OUT_DIR}/structures"            # downloaded mmCIF files
PREPROC_DIR="${OUT_DIR}/preprocessed_data"    # preprocessed .npy outputs
INPUT_CSV="${OUT_DIR}/preprocessing_input.csv"
PATCHED_PREPROCESS="${OUT_DIR}/preprocess_dataset.local.py"

N_DOWNLOAD="${N_DOWNLOAD:-8}"
N_PREPROC="${N_PREPROC:-8}"
SPLITS="${SPLITS:-train,valid,test}"          # comma-separated: train,valid,test

mkdir -p "${STRUCT_DIR}" "${PREPROC_DIR}"

echo "[1/2] Collect PDB IDs from splits..."
PDB_IDS_FILE="${OUT_DIR}/pdb_ids.txt"
python - <<PY
import json, os
repo = r"""${REPO_ROOT}"""
out = r"""${PDB_IDS_FILE}"""
splits = r"""${SPLITS}""".split(",")      #  "train,valid,test" -> ["train", "valid", "test"]
files = [os.path.join(repo, "splits", f"design_{s.strip()}.json") for s in splits if s.strip()]
ids = []
for fp in files:
    with open(fp) as f:
        ids.extend(json.load(f))
ids = sorted(set([x.strip().lower() for x in ids if x and isinstance(x, str)]))
with open(out, "w") as f:
    f.write("\n".join(ids) + ("\n" if ids else ""))
print(f"{len(ids)} ids -> {out}")
PY
# Gives us a file with all PDB IDs (one per line) that we need to download and preprocess.

echo "[1/2] Download mmCIF(.gz) from RCSB (parallel=${N_DOWNLOAD})..."
download_one() {
  local id="$1"
  local sub="${id:1:2}"
  local outdir="${STRUCT_DIR}/${sub}"
  local out="${outdir}/${id}.cif.gz"
  mkdir -p "${outdir}"
  if [[ -s "${out}" ]]; then
    return 0
  fi
  local url="https://files.rcsb.org/download/${id^^}.cif.gz"
  # retries help when downloading thousands of files; add timeouts to avoid hangs
  curl --http1.1 -fsSL --retry 5 --retry-delay 2 \
    --connect-timeout 20 --max-time 120 \
    "${url}" -o "${out}"
}
export -f download_one
export STRUCT_DIR

# xargs parallel download
cat "${PDB_IDS_FILE}" | xargs -n 1 -P "${N_DOWNLOAD}" bash -lc 'download_one "$0"' 

echo "Write ${INPUT_CSV} ..."
python - <<PY
import csv, datetime as dt, os
ids_path = r"""${PDB_IDS_FILE}"""
struct_dir = r"""${STRUCT_DIR}"""
out_csv = r"""${INPUT_CSV}"""
today = dt.date.today().isoformat()

with open(ids_path) as f:
    ids = [x.strip() for x in f if x.strip()]

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id","structure_path","date","dataset_name","ppm_paths"])
    w.writeheader()
    for pid in ids:
        subdir = pid[1:3]
        w.writerow({
            "id": pid,
            "structure_path": os.path.join(struct_dir, subdir, f"{pid}.cif.gz"),
            "date": today,
            "dataset_name": "rcsb_from_splits",
            "ppm_paths": "[]",
        })
print(out_csv)
PY

echo "[2/2] Run NA-MPNN preprocessing (parallel workers=${N_PREPROC})..."
# Patch author hard-coded path "/home/akubaney/projects/na_mpnn" -> this repo root
REPO_ESCAPED="$(printf '%s\n' "${REPO_ROOT}" | sed 's/[&/]/\\&/g')"
sed "s#/home/akubaney/projects/na_mpnn#${REPO_ESCAPED}#g" \
  "${REPO_ROOT}/data/preprocess_dataset.py" > "${PATCHED_PREPROCESS}"

# Use modulo/remainder sharding built into preprocess_dataset.py
pids=()
for r in $(seq 0 $((N_PREPROC - 1))); do
  python "${PATCHED_PREPROCESS}" "${INPUT_CSV}" "${PREPROC_DIR}" "${N_PREPROC}" "${r}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}"; done

echo "Done."
echo "Structures:   ${STRUCT_DIR}"
echo "Preprocessed: ${PREPROC_DIR}"
echo "Input CSV:    ${INPUT_CSV}"
