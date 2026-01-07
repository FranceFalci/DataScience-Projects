import os
import mlflow
import torch
from ultralytics import YOLO, settings

# 1. experiment config
CONFIG = {
    "epochs": 100,
    "batch_size": 64,
    "imgsz": 224,
    "patience": 10,
    "task": "classify",
    "dataset_version": "v1.0.0_training_20251121",
    "experiment_name": "Radiography_Evaluation_Experiments"
}

# 2. models
MODELS_TO_TEST = ["yolo11n-cls", "yolo11m-cls", "yolo11x-cls"]

# 3. paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(CURRENT_DIR, "pretrained_models")
DATA_PATH = os.path.join(CURRENT_DIR, "../../../../../datasets/prueba/split_data/")
LOCAL_RUNS_DIR = os.path.join(CURRENT_DIR, "runs/YOLO")

# 4. MLFlow config
mlflow.set_tracking_uri(f"file:{os.path.join(CURRENT_DIR, 'mlruns')}")
settings.update({"mlflow": True, "runs_dir": LOCAL_RUNS_DIR})


def run_benchmark(model_name):
    print(f"\n🔬 Evaluando candidato: {model_name}")

    model_path = os.path.join(PRETRAINED_DIR, f"{model_name}.pt")

    # load model
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return

    model = YOLO(model_path).to('cuda')

    # MLFlow
    os.environ["MLFLOW_EXPERIMENT_NAME"] = CONFIG["experiment_name"]
    os.environ["MLFLOW_RUN_NAME"] = f"Benchmark_{model_name}"

    # train
    model.train(
        data=DATA_PATH,
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch_size"],
        imgsz=CONFIG["imgsz"],
        patience=CONFIG["patience"],
        task=CONFIG["task"],
        project=LOCAL_RUNS_DIR,
        name=model_name,
        exist_ok=True
    )

    # log
    if mlflow.active_run():
        with mlflow.active_run():
            mlflow.log_param("dataset_version", CONFIG["dataset_version"])
            mlflow.set_tag("stage", "benchmark_comparison")

    # clear GPU
    del model
    torch.cuda.empty_cache()


def main():
    print(f"🚀 Iniciando Benchmark de Modelos.")

    for model_name in MODELS_TO_TEST:
        try:
            run_benchmark(model_name)
        except Exception as e:
            print(f"⚠️ Error en {model_name}: {e}")

    print("\n🏆 Benchmark finalizado. Ejecuta 'mlflow ui' para seleccionar el ganador.")


if __name__ == '__main__':
    main()

"""
usage:
# move to ingeniia_services
dvc pull datasets/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121.dvc
python src/training/train/experiment_benchmark.py
"""
