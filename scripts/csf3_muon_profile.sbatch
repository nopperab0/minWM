#!/bin/bash --login
#SBATCH --job-name=wan0-muon-nsys
#SBATCH --partition=gpuA
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_logs/muon-profile-%j.out
#SBATCH --error=slurm_logs/muon-profile-%j.err

set -euo pipefail
cd "${MINWM_ROOT:-$HOME/scratch/minWM-20261102-new}"
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/26.9.4-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/scratch/minwm-conda-20261102-new"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FR_BUFFER_SIZE=2000
export MINWM_MUON_NVTX=1
RESULTS="$PWD/outputs/muon-nsys-$SLURM_JOB_ID"
mkdir -p "$RESULTS"
python -m pip freeze > "$RESULTS/environment.txt"
nvidia-smi > "$RESULTS/nvidia-smi.txt"
nvidia-smi topo -m > "$RESULTS/topology.txt"
nsys --version > "$RESULTS/nsys-version.txt"
sha256sum minwm/engine/optim/*muon*.py tools/profile_stage0_muon.py > "$RESULTS/source.sha256"
python -m py_compile minwm/engine/optim/owner_muon.py tools/profile_stage0_muon.py
torchrun --standalone --nproc_per_node=4 tools/check_owner_muon.py \
    --device cuda --output "$RESULTS/update-equivalence.json"

for mode in baseline owner; do
    export NSYS_OUTPUT_DIR="$RESULTS"
    export NSYS_RUN_NAME="$mode"
    torchrun --standalone --no-python --nproc_per_node=4 bash tools/profile_nsys.sh \
        python tools/profile_stage0_muon.py --mode "$mode" --output "$RESULTS" \
        --warmup 3 --steps 5 data.dataset.data_path=dataset/benchmark160/data
done

for trace in "$RESULTS"/*.nsys-rep; do
    nsys export --type=sqlite --output="${trace%.nsys-rep}.sqlite" "$trace"
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
        --format csv --output "${trace%.nsys-rep}.stats" "$trace"
done
python tools/summarize_muon_nsys.py "$RESULTS"
echo "PROFILE_COMPLETE $RESULTS"
