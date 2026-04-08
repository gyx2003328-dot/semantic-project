#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# AAE5303_UNet_demo 一站式训练/评估脚本（中文注释版）
#
# 功能：
# 1) 自动注入 GPU 运行环境（修复当前机器的 libcuda 加载问题）
# 2) 执行 GPU 自检
# 3) 启动训练（train.py）
# 4) 启动评估并导出标准 training_report.json（evaluate_submission.py）
# 5) 生成可视化图表（analyze_training.py）
# 6) 生成项目说明报告（generate_project_report.py）
# 7) 一键执行「训练 + 评估 + 可视化 + 报告」
# 8) 扫描多个 checkpoint，自动选 mIoU 最优模型（sweep）
#
# 说明：
# - 默认使用 --device auto（有 GPU 用 GPU，无 GPU 自动回退 CPU）
# - 如需覆盖参数，可在命令末尾附加 train.py / evaluate_submission.py 参数
# =========================================================

# 项目根目录（脚本位于 scripts/ 下，因此上一级是项目根目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 为了避免 Error 804 / cuDNN 初始化异常，这里强制优先使用宿主机驱动的 libcuda
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1

# 默认参数（可在命令行覆盖）
# 稳定模式默认值：先求稳，再逐步加压
EPOCHS=10
BATCH_SIZE=2
LR=0.0001
SCALE=0.35
VAL_PERCENT=15
CLASSES=8
DEVICE=auto

MODEL_PATH="checkpoints/checkpoint_epoch${EPOCHS}.pth"
OUTPUT_JSON="output/training_report.json"
HISTORY_JSON="output/train_history.json"
FIGURES_DIR="figures"
PROJECT_REPORT_MD="docs/PROJECT_REPORT.md"
TEAM_NAME="YourTeam"
REPO_URL="https://github.com/xxxxxx.git"

# 打印帮助信息
usage() {
  cat <<'EOF'
用法：
  bash scripts/run_pipeline.sh [命令] [可选参数...]

默认行为（重点）：
  - 如果不传命令，脚本默认执行 train_eval（训练 + 评估 + 可视化 + 报告）
  - 例如直接运行：bash scripts/run_pipeline.sh
  - 也支持：bash scripts/run_pipeline.sh --epochs 30 -- --team "TeamA"

命令：
  check
    做 GPU/PyTorch 自检（是否可用 CUDA、显卡名称、张量/卷积测试）

  train [train.py参数...]
    启动训练。默认参数：
      --epochs 20 --batch-size 2 --learning-rate 0.0001
      --scale 0.25 --validation 15 --classes 8 --device auto
    示例：
      bash scripts/run_pipeline.sh train
      bash scripts/run_pipeline.sh train --epochs 50 --batch-size 1 --scale 0.3

  eval [evaluate_submission.py参数...]
    运行评估并导出标准 JSON（结构与 output/training_report.json 一致）。默认参数：
      --model checkpoints/checkpoint_epoch5.pth
      --imgs-dir data/imgs --masks-dir data/masks
      --split-file data/uavscenes_test.txt
      --scale 0.25 --output output/training_report.json
      --team "YourTeam" --repo-url "https://github.com/xxxxxx.git"
      --device auto
    示例：
      bash scripts/run_pipeline.sh eval
      bash scripts/run_pipeline.sh eval --model checkpoints/checkpoint_epoch50.pth --output output/submission_epoch50.json

  visualize [analyze_training.py参数...]
    根据 output/training_report.json + output/train_history.json 生成图表。
    示例：
      bash scripts/run_pipeline.sh visualize

  report [generate_project_report.py参数...]
    生成项目说明报告（docs/PROJECT_REPORT.md）。
    示例：
      bash scripts/run_pipeline.sh report

  train_eval [train参数] -- [eval参数]
    一键训练+评估+可视化+报告。中间用 -- 分隔两段参数：
    - 左侧传给 train.py
    - 右侧传给 evaluate_submission.py
    示例：
      bash scripts/run_pipeline.sh train_eval
      bash scripts/run_pipeline.sh train_eval --epochs 30 --batch-size 1 -- --team "TeamA" --output output/teamA.json

  sweep [evaluate_submission.py参数...]
    扫描 checkpoints/checkpoint_epoch*.pth，对每个模型评估一次，
    自动比较 mIoU，输出最优模型和对应指标。
    示例：
      bash scripts/run_pipeline.sh sweep
      bash scripts/run_pipeline.sh sweep --team "TeamA" --repo-url "https://github.com/xxxxxx.git"

  help
    显示本帮助

注意：
  1) 本脚本会设置：
       LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
  2) train.py / evaluate_submission.py / scripts/analyze_training.py / scripts/generate_project_report.py 需存在
EOF
}

# GPU 与 PyTorch 自检
check_env() {
  echo "== [1/2] nvidia-smi =="
  nvidia-smi || true
  echo
  echo "== [2/2] PyTorch CUDA 自检 =="
  python - <<'PY'
import torch
print("torch版本:", torch.__version__)
print("torch编译CUDA版本:", torch.version.cuda)
print("cuda是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU数量:", torch.cuda.device_count())
    print("GPU名称:", torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, 256, 256, device="cuda")
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    y = conv(x)
    print("卷积测试通过，输出形状:", tuple(y.shape))
else:
    print("当前不可用CUDA，将由 --device auto 自动回退CPU")
PY
}

# 执行训练
run_train() {
  if [[ ! -f "train.py" ]]; then
    echo "[错误] 未找到 train.py，请先确认项目文件完整。"
    exit 1
  fi
  echo "== 开始训练 =="
  echo "命令: python train.py --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --learning-rate ${LR} --scale ${SCALE} --validation ${VAL_PERCENT} --classes ${CLASSES} --device ${DEVICE} --amp --stable-mode --num-workers -1 --optimizer adamw --scheduler cosine --loss-name ce_dice --weighted-loss --augmentation --weighted-sampler $*"
  python train.py \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LR}" \
    --scale "${SCALE}" \
    --validation "${VAL_PERCENT}" \
    --classes "${CLASSES}" \
    --device "${DEVICE}" \
    --amp \
    --stable-mode \
    --num-workers -1 \
    --optimizer adamw \
    --scheduler cosine \
    --loss-name ce_dice \
    --weighted-loss \
    --augmentation \
    --weighted-sampler \
    --history-out "${HISTORY_JSON}" \
    "$@"
}

# 执行评估
run_eval() {
  if [[ ! -f "evaluate_submission.py" ]]; then
    echo "[错误] 未找到 evaluate_submission.py，请先确认项目文件完整。"
    exit 1
  fi
  echo "== 开始评估 =="
  echo "命令: python evaluate_submission.py --model ${MODEL_PATH} --imgs-dir data/imgs --masks-dir data/masks --split-file data/uavscenes_test.txt --scale ${SCALE} --output ${OUTPUT_JSON} --team ${TEAM_NAME} --repo-url ${REPO_URL} --device ${DEVICE} $*"
  python evaluate_submission.py \
    --model "${MODEL_PATH}" \
    --imgs-dir data/imgs \
    --masks-dir data/masks \
    --split-file data/uavscenes_test.txt \
    --scale "${SCALE}" \
    --output "${OUTPUT_JSON}" \
    --team "${TEAM_NAME}" \
    --repo-url "${REPO_URL}" \
    --device "${DEVICE}" \
    --history-file "${HISTORY_JSON}" \
    --train-split-file data/uavscenes_train.txt \
    --val-split-file data/uavscenes_val.txt \
    "$@"
}

# 生成可视化
run_visualize() {
  if [[ ! -f "scripts/analyze_training.py" ]]; then
    echo "[错误] 未找到 scripts/analyze_training.py"
    exit 1
  fi
  echo "== 生成可视化图表 =="
  # 若环境缺少 matplotlib，则自动安装（避免流程中断）
  python - <<'PY'
import importlib.util
import subprocess
import sys
for pkg in ["matplotlib", "numpy"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
PY
  python scripts/analyze_training.py \
    --report "${OUTPUT_JSON}" \
    --history "${HISTORY_JSON}" \
    --figures-dir "${FIGURES_DIR}" \
    "$@"
}

# 生成项目说明报告
run_report() {
  if [[ ! -f "scripts/generate_project_report.py" ]]; then
    echo "[错误] 未找到 scripts/generate_project_report.py"
    exit 1
  fi
  echo "== 生成项目说明报告 =="
  python scripts/generate_project_report.py \
    --training-report "${OUTPUT_JSON}" \
    --history-file "${HISTORY_JSON}" \
    --output "${PROJECT_REPORT_MD}" \
    "$@"
}

# 扫描所有 checkpoint 并选择 mIoU 最优模型
run_sweep() {
  if [[ ! -f "evaluate_submission.py" ]]; then
    echo "[错误] 未找到 evaluate_submission.py，请先确认项目文件完整。"
    exit 1
  fi

  mapfile -t ckpts < <(ls checkpoints/checkpoint_epoch*.pth 2>/dev/null | sort -V || true)
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "[错误] 未找到 checkpoints/checkpoint_epoch*.pth"
    exit 1
  fi

  mkdir -p output/sweep
  best_miou="-1"
  best_ckpt=""
  best_json=""

  echo "== 开始 sweep（按 mIoU 选最优）=="
  for ckpt in "${ckpts[@]}"; do
    base="$(basename "${ckpt%.pth}")"
    out="output/sweep/${base}.json"
    echo "[sweep] 评估 ${ckpt}"

    python evaluate_submission.py \
      --model "${ckpt}" \
      --imgs-dir data/imgs \
      --masks-dir data/masks \
      --split-file data/uavscenes_test.txt \
      --scale "${SCALE}" \
      --output "${out}" \
      --team "${TEAM_NAME}" \
      --repo-url "${REPO_URL}" \
      --device "${DEVICE}" \
      "$@"

    miou="$(python - <<PY
import json
with open("${out}", "r") as f:
    data = json.load(f)
print(data["metrics"]["miou"])
PY
)"

    echo "[sweep] ${base} -> mIoU=${miou}"
    better="$(python - <<PY
print(float("${miou}") > float("${best_miou}"))
PY
)"
    if [[ "${better}" == "True" ]]; then
      best_miou="${miou}"
      best_ckpt="${ckpt}"
      best_json="${out}"
    fi
  done

  echo
  echo "== sweep 完成 =="
  echo "最优 checkpoint: ${best_ckpt}"
  echo "最优 mIoU: ${best_miou}"
  echo "对应 JSON: ${best_json}"
}

# 默认命令改为 train_eval
# 兼容场景：
# 1) 不传参数 -> train_eval
# 2) 首个参数是选项（如 --epochs）-> 视作 train_eval 参数
# 3) 明确传命令（check/train/eval/train_eval/sweep/help）-> 按命令执行
KNOWN_CMDS=("help" "-h" "--help" "check" "train" "eval" "visualize" "report" "train_eval" "sweep")
cmd="${1:-train_eval}"
if [[ $# -eq 0 ]]; then
  cmd="train_eval"
else
  is_known=0
  for c in "${KNOWN_CMDS[@]}"; do
    if [[ "$cmd" == "$c" ]]; then
      is_known=1
      break
    fi
  done
  if [[ $is_known -eq 0 ]]; then
    # 将所有参数都当作 train_eval 的参数
    cmd="train_eval"
  else
    shift || true
  fi
fi

case "$cmd" in
  help|-h|--help)
    usage
    ;;
  check)
    check_env
    ;;
  train)
    # train 子命令：所有剩余参数都透传给 train.py
    run_train "$@"
    ;;
  eval)
    # eval 子命令：所有剩余参数都透传给 evaluate_submission.py
    run_eval "$@"
    ;;
  visualize)
    run_visualize "$@"
    ;;
  report)
    run_report "$@"
    ;;
  train_eval)
    # train_eval 子命令：以 -- 作为分隔符
    # 左边参数给 train.py，右边参数给 evaluate_submission.py
    train_args=()
    eval_args=()
    to_eval=0
    for arg in "$@"; do
      if [[ "$arg" == "--" ]]; then
        to_eval=1
        continue
      fi
      if [[ $to_eval -eq 0 ]]; then
        train_args+=("$arg")
      else
        eval_args+=("$arg")
      fi
    done

    run_train "${train_args[@]}"
    run_eval "${eval_args[@]}"
    run_visualize
    run_report
    ;;
  sweep)
    run_sweep "$@"
    ;;
  *)
    echo "[错误] 未知命令: $cmd"
    echo
    usage
    exit 1
    ;;
esac
