from src.pipeline import load_dataset, train_models

if __name__ == "__main__":
    data = load_dataset()
    artifact, report, _ = train_models(data)
    print("\nGridPulse training completed.")
    print(f"Usable records: {report['total_rows']:,}")
    print(f"Predictors: {report['feature_count']}")
    print(f"Selected model: {artifact['model_name']}")
    print(f"Decision threshold: {artifact['threshold']:.3f}")
    print(f"Test F1-score: {report['selected_metrics']['f1']:.3f}")
    print("Saved: models/best_model.joblib")
    print("Saved: outputs/metrics.json")
    print("Saved: outputs/misclassified_examples.csv")
