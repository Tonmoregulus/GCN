import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data import Dataset
from src.utils import config2string
from src.utils import compute_accuracy
from layers import GCN, Classifier

class embedder:
    def __init__(self, args):
        self.args = args

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
        # dataset
        self.data = Dataset(root=args.root, dataset=args.dataset)[0].to(self.device)
        
        # Encoder
        self.hidden_layers = eval(args.layers)
        input_size = self.data.x.size(1)
        rep_size = self.hidden_layers[-1]

        self.unique_labels = self.data.y.unique()
        self.num_classes = len(self.unique_labels)

        self.encoder = GCN([input_size] + self.hidden_layers)
        self.decoder = InnerProductDecoder(dropout = 0.5)
        self.classifier = Classifier(rep_size, self.num_classes)

        # For Evaluation
        self.best_val = 0 
        self.epoch_list = []
        
        self.train_accs = [] ; self.valid_accs = [] ; self.test_accs = [] 
        self.running_train_accs = [] ; self.running_valid_accs = [] ; self.running_test_accs = []

        
    def evaluate(self, batch_data, st):

        # Classifier Accuracy
        self.model.eval()
        _, preds = self.model.cls(batch_data)
        
        train_acc, val_acc, test_acc = compute_accuracy(preds, batch_data.y, self.train_mask, self.val_mask, self.test_mask)
        self.running_train_accs.append(train_acc) ; self.running_valid_accs.append(val_acc) ; self.running_test_accs.append(test_acc)        

        if val_acc > self.best_val:
            self.best_val = val_acc
            self.cnt = 0
        else:
            self.cnt += 1
        
        st += '| train_acc: {:.2f} | valid_acc : {:.2f} | test_acc : {:.2f} '.format(train_acc, val_acc, test_acc)
        print(st)
        
    def save_results(self, fold):

        train_acc, val_acc, test_acc = torch.tensor(self.running_train_accs), torch.tensor(self.running_valid_accs), torch.tensor(self.running_test_accs)        
        selected_epoch = val_acc.argmax()
        
        best_train_acc = train_acc[selected_epoch]
        best_val_acc = val_acc[selected_epoch]
        best_test_acc = test_acc[selected_epoch]

        self.epoch_list.append(selected_epoch.item())
        self.train_accs.append(best_train_acc) ; self.valid_accs.append(best_val_acc) ; self.test_accs.append(best_test_acc)

        if fold+1 != self.args.folds:
            self.running_train_accs = [] ; self.running_valid_accs = [] ; self.running_test_accs = []
            
            self.cnt = 0
            self.best_val = 0


    def summary(self):

        train_acc_mean = torch.tensor(self.train_accs).mean().item()
        val_acc_mean = torch.tensor(self.valid_accs).mean().item()
        test_acc_mean = torch.tensor(self.test_accs).mean().item()
        print("** | train acc : {:.2f} | valid acc : {:.2f} | test acc : {:.2f} |  **\n".format(
                    train_acc_mean, val_acc_mean, test_acc_mean))



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj        