import torch
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

torch.manual_seed(42)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

allteams = []

with open(file="2025_LoL_esports_match_data_from_OraclesElixir.csv", mode="r", newline="", encoding="utf-8") as f:
    data = csv.DictReader(f)
    datalist = list(data)

h2h = defaultdict(lambda: [0, 0])

for row in datalist:
    if row["participantid"] == "100" or row["participantid"] == "200":
        allteams.append(row["teamname"])

matchCount = len(allteams)//2
finishedteams = np.unique(np.array(allteams)).reshape(-1)
teamsnametoids = {}
teamsidtonames = {}
for idx, name in enumerate(finishedteams):
    teamsnametoids[name] = idx
    teamsnametoids[idx] = name
twoteamfeats = 4
teamsdata = torch.zeros(matchCount, twoteamfeats)
resultsdata = torch.zeros(matchCount, 1)

loopnum = 0

T1 = False
T2 = False
x = None
y = None
z = None

for row in datalist:
    if row["participantid"] == "100":
        x = teamsnametoids[row["teamname"]]
        team1 = row["teamname"]
        if row["result"] == "1":
            z = 1
        else:
            z = 0
        T1 = True

    if row["participantid"] == "200":
        y = teamsnametoids[row["teamname"]]
        team2 = row["teamname"]
        T2 = True

    if T1 == True and T2 == True:
        sortedmatchteams = tuple(sorted([team1, team2]))
        t1_wins, t2_wins = h2h[sortedmatchteams]
        matchupcount = t1_wins + t2_wins
        t1_wr = t1_wins / matchupcount if matchupcount > 0 else 0
        t2_wr = t2_wins / matchupcount if matchupcount > 0 else 0
        winrates = (t1_wr, t2_wr)
        if sortedmatchteams[0] != team1:
            winrates = (t2_wr, t1_wr)
        numpyarray = np.concatenate((y, x, winrates), axis=None, dtype=np.float32)
        teamsdata[loopnum] = torch.from_numpy(numpyarray)
        resultsdata[loopnum] = z
        loopnum += 1

        if sortedmatchteams[0] == team1:
            h2h[sortedmatchteams][0] += z
            h2h[sortedmatchteams][1] += 1 - z
        else:
            h2h[sortedmatchteams][1] += z
            h2h[sortedmatchteams][0] += 1 - z

        T1 = False
        T2 = False


X_data = teamsdata
y_data = resultsdata

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=len(finishedteams), embedding_dim=16)
        self.passthrough = nn.Sequential(
            nn.Linear(34, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )

    def forward(self, x):
        modelteams, modelwr = x[:, :2], x[:, 2:]
        modelteams = torch.tensor(data=modelteams, dtype=torch.long)
        emb = self.embedding(modelteams)
        emb = emb.view(128, 32)
        x = torch.cat((emb, modelwr), dim=1)
        return self.passthrough(x)

model1 = Net()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model1.parameters(), lr=0.0005, weight_decay=0.0005)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()

    accuracy = correct / len(y_pred) * 100
    return accuracy

epochs = 100
trainloss = []
testloss = []
trainaccuracy = []
testaccuracy = []
epochcount = []

for epoch in range(epochs):
    model1.train()
    train_batch_preds = []
    train_batch_labels = []
    train_batch_loss = 0

    for X_batch, y_batch in train_loader:
        y_logits = model1(X_batch)
        loss = loss_fn(y_logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_batch_loss += loss.item() * len(y_batch)
        train_batch_preds.append(torch.sigmoid(y_logits))
        train_batch_labels.append(y_batch)

    train_epoch_preds = torch.cat(train_batch_preds)
    train_epoch_labels = torch.cat(train_batch_labels)
    train_epoch_loss = train_batch_loss / len(train_loader.dataset)
    train_epoch_acc = accuracy_fn(y_true=train_epoch_labels, y_pred=torch.round(train_epoch_preds))

    trainloss.append(train_epoch_loss)
    trainaccuracy.append(train_epoch_acc)

    model1.eval()
    with (torch.inference_mode()):
        test_batch_preds = []
        test_batch_labels = []
        test_batch_loss = 0

        for X_batch, y_batch in test_loader:
            test_logits = model1(X_batch)
            test_loss = loss_fn(test_logits, y_batch)

            test_batch_loss += test_loss.item() * len(y_batch)
            test_batch_labels.append(y_batch)
            test_batch_preds.append(torch.sigmoid(test_logits))


        test_epoch_preds = torch.cat(test_batch_preds)
        test_epoch_labels = torch.cat(test_batch_labels)
        test_epoch_loss = test_batch_loss / len(test_loader.dataset)
        test_epoch_acc = accuracy_fn(y_true=test_epoch_labels, y_pred=torch.round(test_epoch_preds))

        testloss.append(test_epoch_loss)
        testaccuracy.append(test_epoch_acc)
        epochcount.append(epoch)

        if epoch == 99:
            ax[0].plot(epochcount, [v for v in trainloss], label="Training Loss")
            ax[0].plot(epochcount, [v for v in testloss], label="Test Loss")
            ax[0].set_title("Training and Test Loss")
            ax[0].set_ylabel("Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].legend(prop={'size': 14})

            ax[1].plot(epochcount, [v for v in trainaccuracy], label="Training Accuracy")
            ax[1].plot(epochcount, [v for v in testaccuracy], label="Test Accuracy")
            ax[1].set_title("Training and Test Accuracy")
            ax[1].set_ylabel("Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].legend(prop={'size': 14})


            match1 = torch.cat(torch.split(X_test[0], 1, dim=0))
            match2 = torch.cat(torch.split(X_test[1], 1, dim=0))
            match3 = torch.cat(torch.split(X_test[2], 1, dim=0))
            test_match1 = encoder.inverse_transform(match1[:-2].view(2, -1)).reshape(2)
            test_match2 = encoder.inverse_transform(match2[:-2].view(2, -1)).reshape(2)
            test_match3 = encoder.inverse_transform(match3[:-2].view(2, -1)).reshape(2)

            test_match1_string = f"{test_match1[0]} vs {test_match1[1]}"
            test_match2_string = f"{test_match2[0]} vs {test_match2[1]}"
            test_match3_string = f"{test_match3[0]} vs {test_match3[1]}"

            test_graph_results = torch.sigmoid(torch.cat((test_logits[0], test_logits[1], test_logits[2]))) #.transpose(0,1).squeeze()

            ax[2].bar([test_match1_string, test_match2_string, test_match3_string], test_graph_results)
            ax[2].set_title("Test Matches And Predictions After Training")
            ax[2].set_ylim(bottom=0, top=1)
            ax[2].set_ylabel("Chance that Team 1 wins")
            ax[2].set_xlabel("Test Matches")

            plt.show()

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_Saved_Net_Model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model1.state_dict(), f=MODEL_SAVE_PATH)