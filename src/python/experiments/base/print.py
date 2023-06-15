def print_info(
    experiment_name: str,
    algorithm: str,
    environment_name: str,
    seed: int,
    train: bool = True,
):
    print(f"-------- {experiment_name} --------")
    if train:
        print(
            f"Training {algorithm} on {environment_name} with seed {seed}..."
        )
    else:
        print(
            f"Evaluating {algorithm} on {environment_name} with seed {seed}..."
        )
