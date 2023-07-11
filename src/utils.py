import argparse

def get_args():

    parser = argparse.ArgumentParser("""Train model for Pokemon dataset""")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="batch size of dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="number of epochs")
    parser.add_argument("--log_path", "-l", type=str, default="../tensorboard/pokemon", help="path to tensorboard")
    parser.add_argument("--save_path", "-s", type=str, default="../trained_models/pokemon", help="path to trained models")
    parser.add_argument("--load_checkpoint", "-m", type=str, default="../trained_models/pokemon/last.pt", help="path to checkpoint to be loaded")
    args = parser.parse_args()
    return args

