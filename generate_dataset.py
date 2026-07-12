from src.circuit_generator import generate_dataset_batches

generate_dataset_batches(
    output_directory="generated_batches",
    total_circuits=50000,
    batch_size=1000,
    random_seed=42
)
