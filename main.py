from mlb_predictor.data_loader import load_data

if __name__ == "__main__":
    data = load_data("statsapi")
    print(data.head())  # Will also trigger logging