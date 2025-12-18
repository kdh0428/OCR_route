# OCR Route: Uncertainty-Triggered OCR Routing with Qwen2-VL

## Overview
This project implements an uncertainty-aware routing pipeline that augments multimodal models—tested with Qwen/Qwen2-VL-7B-Instruct, guoxy25/Ocean-OCR, and llava-hf/llava-1.5-7b-hf—with optional OCR assistance. The system first answers vision-language questions directly with the base model, then measures answer uncertainty via token-level entropy and logit margin statistics. When uncertainty exceeds configurable thresholds, the pipeline falls back to external OCR engines (EasyOCR or Tesseract) to extract auxiliary text and performs a second pass before selecting the final answer.

## Installation
1. (Optional) create and activate a virtual environment for Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   PaddleOCR는 실행 환경에 따라 `paddlepaddle` 추가 설치가 요구될 수 있습니다. (예: CPU 환경)

## Quick Start
Process a single image-question pair with the default Qwen2-VL backbone:
```bash
python -m ocr_route.cli --image examples/demo.jpg --question "What does the sign say?"
```
To swap in Ocean-OCR:
```bash
python -m ocr_route.cli \
  --model-id guoxy25/Ocean-OCR \
  --image examples/demo.jpg \
  --question "What does the sign say?"
```
Qwen3-VL도 동일하게 사용할 수 있습니다:
```bash
python -m ocr_route.cli \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --image examples/demo.jpg \
  --question "What does the sign say?"
```

## Routing Logic
- The first pass runs Qwen2-VL without OCR assistance.
- Token entropies and logit margins measure uncertainty.
- Routing triggers when `mean_entropy` exceeds `--entropy-th` or `min_margin` drops below `--margin-th`.
- On trigger (or `--always-ocr`), an OCR engine extracts text that is injected into the prompt for a second pass. 기본 엔진은 PaddleOCR이며 `--ocr-engine easyocr|tesseract`로 변경할 수 있습니다.
- The final answer is chosen by comparing uncertainty metrics; lower entropy or higher margin wins.

### Tuning Tips
- Lower `--entropy-th` or raise `--margin-th` to invoke OCR more often.
- Use `--no-routing` to disable OCR entirely, or `--always-ocr` to force it every time.
- Adjust `--min-pixels` and `--max-pixels` if you need tighter control over image resolution for the processor. 기본값으로 `--max-pixels`가 512,000으로 설정되어 있어 720p 수준으로 다운스케일링됩니다.

### Threshold Calibration & Risk Scoring
- `scripts/tune_thresholds.py`를 활용하면 엔트로피/마진 분포 기반으로 `(entropy_a, entropy_b, margin_tau)` 조합을 자동으로 찾을 수 있고, 커버리지 60–80% 범위에서 F1이 최대가 되는 임계치를 손쉽게 캘리브레이션할 수 있습니다.
- 엔트로피와 마진을 결합한 위험 점수 `risk = alpha * mean_entropy - beta * min_margin`를 사용하고 싶다면 `--risk-alpha`, `--risk-beta`, `--risk-tau` 플래그를 함께 넘겨 추가적인 트리거 조건을 둘 수 있습니다.
- OCR 결과는 정규화되어 프롬프트에 삽입되며, “OCR text (may contain errors); verify against the image before finalizing” 안내 문구가 자동으로 붙어 모델이 OCR 노이즈를 감안하도록 돕습니다.

### Uncertainty Calibration (Step 1)
1. JSONL 로그를 준비합니다(`out1_entropy`, `out1_margin`, `label_call_ocr_good` 포함).
2. 온도 스케일링 실행:
   ```bash
   python scripts/calibrate_uncertainty.py \
     --train data/train.jsonl --valid data/valid.jsonl \
     --use_margin true --w1 1.0 --w2 1.0 \
     --out artifacts/calibrator.json \
     --report_dir reports/calibration
   ```
   - `artifacts/calibrator.json`에 추정된 온도(`T`)와 사용 피처 목록이 저장됩니다.
   - `reports/calibration/`에는 ECE 전/후 CSV 및 요약 이미지가 생성됩니다.

### Feature Vectorization (Step 2)
- `src/features/jsonl.py`: JSONL 읽기 및 안전 접근(`safe_get`) 헬퍼.
- `src/features/build.py`: `build_example()`로 JSON 레코드를 피처 벡터와 레이블로 변환. 캘리브레이터에서 가져온 온도 값을 적용한 `calib_entropy`, `calib_neg_margin` 등 기본 피처가 포함됩니다.
- 원하는 피처 구성은 `artifacts/feature_spec.json` 형태로 작성 후 `build_example(row, calibrator, spec)` 호출 시 자동 적용됩니다.

## Batch Execution
Run a CSV batch (columns: `image`, `question`) and save JSONL results:
```bash
python -m ocr_route.cli --csv examples/demo.csv --out outputs/results.jsonl
```
Progress is tracked with `tqdm`, and each record includes uncertainty metrics and latency timings.

## Dataset Evaluation
`scripts/run_dataset_eval.py`는 여러 OCR 데이터 소스를 지원하며, 선택한 실험군(베이스라인, 항상 OCR, 엔트로피 기반 라우팅, 온도 스케일링 캘리브레이션, 마진 어블레이션 등)을 일괄 실행하는 sweep 모드를 제공합니다. 각 실험은 아래 지표를 JSONL과 요약 JSON에 기록하고, W&B가 활성화되어 있다면 동일한 태그(`experiment/<name>/...`)로 실시간 로깅됩니다.

- 정밀도 지표: CER, WER, Exact Match, F1, Precision/Recall, ROUGE-L
- 라우팅 지표: OCR 호출 비율, 평균 지연시간(sec/sample), 엔트로피 분포(트리거 vs 비트리거)
- 캘리브레이션 지표: Expected Calibration Error(ECE), Brier score, confidence histogram
- Pareto 분석을 위한 `latency_vs_quality` 테이블과 risk-coverage 포인트(coverage=OCR 비율, risk=1-accuracy)

대표적인 sweep 실행 예시:
```bash
python3 scripts/run_dataset_eval.py \
  --dataset cc-ocr \
  --config multi_scene_ocr \
  --split test \
  --question-template "Please transcribe all visible text in this chart." \
  --out outputs/cc_ocr_results_test.jsonl \
  --model-id Qwen/Qwen2-VL-7B-Instruct \
  --max-pixels 512000 \
  --ocr-engine easyocr \
  --experiment-set paper \
  --enable-wandb \
  --wandb-project ocr-routing \
  --wandb-entity your-team \
  --wandb-tag cc-ocr \
  --wandb-tag sweep
```
`--experiment-set paper`는 다음 실험을 자동으로 포함합니다.

1. Baseline (no OCR)
2. Always-OCR
3. Entropy-only routing: `entropy_th ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}`, `history_k ∈ {0, 3, 5}`
4. Entropy-only + temperature calibration: `temperature_scale ∈ {0.6, 0.7, 0.8}`
5. Margin ablation: `margin_th ∈ {0.5, 1.0, 1.5}` (entropy+margin)와 `margin_only_{…}` 실험(엔트로피 비활성화, 마진만 사용)

각 실험 결과는 `outputs/cc_ocr_results_test__{experiment}.jsonl`로 저장되고, 최종 요약은 표준 출력(JSON)과 `--out` 경로에 기록됩니다. 커스텀 sweep을 구성하고 싶다면 `--entropy-th-grid`, `--history-k-grid`, `--margin-th-grid`, `--temperature-scale-grid`, `--include-baseline`, `--include-always`, `--include-margin-ablation` 등의 옵션을 조합하면 됩니다.

빠른 검증 프리셋(추천 조합: history_k=3, temperature_scale=0.7, margin_th=0.5, ocr_prompt_mode=verify) 예시:
```bash
python3 scripts/run_dataset_eval.py \
  --dataset cc-ocr \
  --config multi_scene_ocr \
  --split test \
  --question-template "Please transcribe all visible text in this chart." \
  --out outputs/cc_ocr_quick_verify.jsonl \
  --model-id Qwen/Qwen2-VL-7B-Instruct \
  --max-pixels 800000 \
  --ocr-engine paddleocr \
  --experiment-set quick_verify
```
`quick_verify`는 baseline(no OCR)과 verify-mode 추천 라우팅 한 쌍만 돌려 빠르게 비교합니다.
다른 실험 세트에 이 조합만 추가하려면 `--include-verify-combo` 플래그를 붙이면 됩니다.

- **TextVQA JSON + 이미지** (로컬 파일 필요):
```bash
scripts/run_dataset_eval.py \
  --dataset textvqa \
  --questions data/textvqa/TextVQA_0.5.1_val.json \
  --images-root data/textvqa/images \
  --image-template "{image_id}.jpg" \
  --out outputs/textvqa_results.jsonl \
  --model-id Qwen/Qwen2-VL-7B-Instruct --max-pixels 512000
```
- **TrainingDataPro/ocr-text-detection-in-the-documents** (Hugging Face Datasets):
  ```bash
scripts/run_dataset_eval.py \
  --dataset trainingdatapro \
  --split train \
  --question-template "Please transcribe all visible text in this document image." \
  --out outputs/trainingdatapro_results.jsonl \
  --model-id Qwen/Qwen2-VL-7B-Instruct --max-pixels 512000
```
- **wulipc/CC-OCR** (Hugging Face Datasets):
  ```bash
scripts/run_dataset_eval.py \
  --dataset cc-ocr \
  --config multi_scene_ocr \
  --split train \
  --question-template "Please transcribe all visible text in this chart." \
  --out outputs/cc_ocr_results.jsonl \
  --model-id Qwen/Qwen2-VL-7B-Instruct --max-pixels 512000
```
Ocean-OCR로 동일 실험을 돌리고 싶다면 `--model-id guoxy25/Ocean-OCR`만 바꿔 주면 됩니다 (필요 시 `--ocr-engine`과 픽셀 스케일도 조정).
Qwen3-VL 실험 역시 `--model-id Qwen/Qwen3-VL-8B-Instruct` 등으로 변경하면 동일하게 수행됩니다.

CC-OCR처럼 긴 텍스트/LaTeX 정답이 많은 경우, 평가 JSONL을 바탕으로 동적 임계치를 찾을 수 있습니다. 예시:
```bash
scripts/tune_thresholds.py \
  --results outputs/cc_ocr_results_300.jsonl \
  --entropy-a 0.3:0.7:0.1 \
  --entropy-b 0.0:0.4:0.1 \
  --margin-tau 0.05:0.20:0.05 \
  --risk-alpha 0.4:0.8:0.1 \
  --risk-beta 0.2:0.6:0.1 \
  --risk-tau 0.02:0.08:0.02 \
  --coverage-min 0.6 --coverage-max 0.8
```
출력된 `(entropy_a, entropy_b, margin_tau, risk_alpha, risk_beta, risk_tau)`는 `python -m ocr_route.cli` 실행 시 `--entropy-a`, `--entropy-b`, `--margin-tau`, `--risk-alpha`, `--risk-beta`, `--risk-tau`, `--history-k` 옵션으로 전달해 길이 보정 + 히스테리시스 + 위험점수 기반 라우팅을 적용할 수 있습니다.

Hugging Face `datasets` 라이브러리가 필요하며, 인증이 요구되는 리포는 `HF_TOKEN` 환경 변수 등을 이용해 권한을 부여해야 합니다. 추가 CLI 옵션(엔트로피 임계치, OCR 엔진 등)은 명령 끝에 이어 붙이면 파이프라인으로 전달됩니다.

## Known Limitations
- Tesseract requires a local installation of the Tesseract binary; configure `TESSDATA_PREFIX` if needed.
- EasyOCR uses GPU automatically when available; CPU-only inference can be slower.
- Large Qwen2-VL models demand substantial GPU memory; adjust `--dtype` and `--max-pixels` for lower resource setups.
- OCR quality greatly influences the second pass; noisy outputs may not improve uncertainty.
LLaVA 1.5 (HF-converted checkpoints) can also be evaluated by switching only the `--model-id` flag:

```bash
python3 scripts/run_dataset_eval.py \
  --dataset cc-ocr \
  --config multi_scene_ocr \
  --split test \
  --limit 300 \
  --question-template "Please transcribe all visible text in this chart." \
  --out outputs/cc_ocr_results_llava15.jsonl \
  --model-id llava-hf/llava-1.5-7b-hf \
  --ocr-engine easyocr
```
