import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="cora", help="cora, citeseer, pubmed, computers, photo")
    
    # masking
    parser.add_argument("--label_rate", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=2)

    # Encoder
    parser.add_argument("--layers", nargs='+', default='[128, 128]', help="The number of units of each layer of the GNN. Default is [256]")
    
    # optimization
    parser.add_argument("--epochs", '-e', type=int, default=1000, help="The number of epochs")    
    parser.add_argument("--lr", '-lr', type=float, default=0.01, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--decay", type=float, default=1e-5, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--patience", type=int, default=200)
        
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--clustering", action='store_true', default=True)
    parser.add_argument("--num_K", type=int, default=200)
    parser.add_argument("--KNNneighbors", type=int, default=30)
    parser.add_argument("--device", '-d', type=int, default=0, help="GPU to use")    
    
    return parser.parse_known_args()[0]
