from pathlib import Path

import pandas as pd

from src.config import load_config


def main():
    cfg = load_config()
    df = pd.read_csv(cfg.paths.data_path)
    print(df.head)
    print("Hello from fraud-serve-bench!")


if __name__ == "__main__":
    main()
