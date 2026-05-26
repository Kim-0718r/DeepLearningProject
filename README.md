# Deep Learning Mini Project

농산물 품목 10종 이미지 분류 프로젝트입니다. 조원들이 만든 코드를 하나의 실행 흐름으로 합쳐, 아래 4가지 실험 조건을 비교합니다.

## 1. 실험 조건

| 실험 | 모델 | 학습 방식 | 목적 |
|---|---|---|---|
| Exp1 | ScratchCNN | 처음부터 학습 | 간단한 CNN 기준 성능 확인 |
| Exp2 | EfficientNet-B0 | 처음부터 학습 | 동일 아키텍처에서 사전학습 없이 학습한 성능 확인 |
| Exp3 | EfficientNet-B0 | Feature Extraction | ImageNet 사전학습 특징 추출기 고정, classifier만 학습 |
| Exp4 | EfficientNet-B0 | Fine-tuning | ImageNet 사전학습 가중치에서 전체 layer 재학습 |

기본 실행 모델은 `src/config.py`의 `MODEL_NAMES`에 정의되어 있습니다.

## 2. 데이터 준비

AI Hub에서 받은 zip 파일들을 프로젝트 루트의 `AI_Project` 폴더에 넣습니다.

```powershell
python prepare_data.py
```

데이터 전처리는 `src/data_preprocessing.py`에서 수행합니다. 기존 `src/Data Preprocessing.py`의 농산물 클래스 매핑, 압축 해제, 클래스별 샘플링, 8:1:1 분할, augmentation 흐름을 학습 코드에서 import 가능한 형태로 정리한 파일입니다.

실행 후 데이터는 아래 구조로 생성됩니다.

```text
data/processed/
  train/
  val/
  test/
```

각 split 안에는 `Apple`, `Cabbage`, `Chinese-cabbage`, `Garlic`, `Mandarine`, `Onion`, `Pear`, `Persimmon`, `Potato`, `Radish` 클래스 폴더가 만들어집니다.

## 3. 실험 실행

```powershell
python run_experiment.py
```

실행하면 각 모델별 학습, 검증, 테스트 평가가 순서대로 진행됩니다.

## 4. 결과 저장 위치

```text
outputs/checkpoints/  # validation accuracy 기준 best checkpoint
outputs/figures/      # loss curve, accuracy curve, confusion matrix, comparison chart
outputs/metrics/      # classification report, confusion pairs, model comparison json
```

## 5. 보고서에 쓸 설명

본 프로젝트는 농산물 품목 10종 이미지 분류 문제를 해결하기 위해 ScratchCNN, 사전학습을 사용하지 않은 EfficientNet-B0, Feature Extraction 방식의 EfficientNet-B0, Fine-tuning 방식의 EfficientNet-B0를 비교하였다.

Feature Extraction 실험에서는 ImageNet으로 사전학습된 EfficientNet-B0의 feature extractor를 고정하고 마지막 classifier layer만 농산물 10개 클래스에 맞게 교체하여 학습하였다. Fine-tuning 실험에서는 ImageNet 사전학습 가중치를 초기값으로 사용하되 전체 layer를 학습 가능하게 두어 농산물 이미지의 색상, 질감, 형태적 특징에 더 적응하도록 하였다.

최종 보고서에서는 네 모델의 Test Accuracy, Loss/Accuracy Curve, Confusion Matrix, 주요 오분류 쌍을 비교하면 된다.
