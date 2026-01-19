import pandas as pd
from pathlib import Path


def main():
    print("Hello There, This is CLI of DataCraft v1")
    print("For yes write Y, for no write N")
    
    path = Path(input("Please enter your file path: ").strip())

    if not path.exists():
        print("Error: File doesn't exist")

    if path.suffix != ".csv":
        print("Error: Only CSV files are supported")

    df = pd.read_csv(path)
    print("Dataset loaded successfully")
    print(df.head())


if __name__ == "__main__":
    main()
