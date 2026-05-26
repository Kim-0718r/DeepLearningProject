from pathlib import Path

import pandas as pd


def get_top_misclassifications(cm, class_names, top_k=10):
    """
    Confusion Matrix에서 오분류가 많이 발생한 클래스 쌍 top-k를 반환하는 함수

    예:
    True: Apple, Predicted: Pear, Count: 12
    """
    mistakes = []

    for true_idx, true_name in enumerate(class_names):
        for pred_idx, pred_name in enumerate(class_names):
            if true_idx == pred_idx:
                continue

            count = int(cm[true_idx, pred_idx])

            if count > 0:
                mistakes.append(
                    {
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "count": count,
                    }
                )

    mistakes = sorted(
        mistakes,
        key=lambda x: x["count"],
        reverse=True,
    )

    return mistakes[:top_k]


def save_top_misclassifications(cm, class_names, save_path, top_k=10):
    """
    오분류 top-k 결과를 CSV 파일로 저장하는 함수
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mistakes = get_top_misclassifications(
        cm=cm,
        class_names=class_names,
        top_k=top_k,
    )

    df = pd.DataFrame(mistakes)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[Saved] Top misclassifications saved to: {save_path}")

    return df


def print_top_misclassifications(cm, class_names, top_k=10):
    """
    오분류 top-k 결과를 콘솔에 출력하는 함수
    """
    mistakes = get_top_misclassifications(
        cm=cm,
        class_names=class_names,
        top_k=top_k,
    )

    print("===== Top Misclassifications =====")

    if len(mistakes) == 0:
        print("No misclassifications found.")
        return

    for idx, item in enumerate(mistakes, start=1):
        print(
            f"{idx}. True: {item['true_class']} "
            f"-> Predicted: {item['predicted_class']} "
            f"({item['count']} samples)"
        )

    print("==================================")


def summarize_model_result(model_name, results):
    """
    보고서 작성용으로 모델 결과를 간단히 요약하는 함수

    results는 evaluate_model()의 반환값을 사용한다.
    """
    report = results["classification_report"]

    summary = {
        "model_name": model_name,
        "test_loss": round(float(results["test_loss"]), 4),
        "test_acc": round(float(results["test_acc"]), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_precision": round(float(report["weighted avg"]["precision"]), 4),
        "weighted_recall": round(float(report["weighted avg"]["recall"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
    }

    return summary


def save_model_summaries(summaries, save_path):
    """
    여러 모델의 결과 요약을 CSV 파일로 저장하는 함수

    summaries 예시:
    [
        summarize_model_result("scratch_cnn", scratch_results),
        summarize_model_result("feature_extraction", fe_results),
        summarize_model_result("fine_tuning", ft_results),
    ]
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(summaries)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[Saved] Model summary saved to: {save_path}")

    return df