import torch
from sklearn.model_selection import train_test_split
from torch import nn
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

encoder = OneHotEncoder()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

with open("data.json", "r") as f:
    data = json.load(f)

allteams = []

for match in data:
    allteams.append(data[match]["Team1"])
    allteams.append(data[match]["Team2"])

finishedteams = np.unique(np.array(allteams)).reshape(-1,1)

teamsdata = torch.zeros(len(data), 6)
resultsdata = torch.zeros(len(data), 1)

encoder.fit(finishedteams)

loopnum = 0

for match in data:
    x = (encoder.transform(np.array([data[match]["Team1"]]).reshape(1,-1))).toarray().reshape(-1)
    y = (encoder.transform(np.array([data[match]["Team2"]]).reshape(1,-1))).toarray().reshape(-1)
    z = (encoder.transform(np.array([data[match]["Winner"]]).reshape(1,-1))).toarray().reshape(-1)

    if np.array_equal(x,z):
        z = 1
    else:
        z = 0

    numpyarray = np.concatenate((x, y) ,dtype=np.float32)
    teamsdata[loopnum] = torch.from_numpy(numpyarray)
    resultsdata[loopnum] = z
    loopnum += 1


X_data = teamsdata
y_data = resultsdata

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

team1 = X_train[:,:3]
team2 = X_train[:,3:]

X_train_flipped = torch.cat((team2, team1), dim=1)
y_train_flipped = (1 - y_train).float()

X_train = torch.cat((X_train, X_train_flipped), dim=0)
y_train = torch.cat((y_train, y_train_flipped), dim=0)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.passthrough = nn.Sequential(
            nn.Linear(6, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.passthrough(x)

model1 = Net()

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()

    accuracy = correct / len(y_pred) * 100
    return accuracy

epochs = 400

trainloss = []
testloss = []
trainaccuracy = []
testaccuracy = []
epochcount = []

for epoch in range(epochs):
    model1.train()

    y_logits = model1(X_train)

    loss = loss_fn(y_logits, y_train)

    train_acc = accuracy_fn(y_true=y_train, y_pred=torch.round(torch.sigmoid(y_logits)))

    trainloss.append(loss)
    trainaccuracy.append(train_acc)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model1.eval()
    with torch.inference_mode():

        test_logits = model1(X_test)
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)

        testloss.append(test_loss)
        testaccuracy.append(test_acc)
        epochcount.append(epoch)

        if epoch == 399:
            ax[0].plot(epochcount, [v.detach().numpy() for v in trainloss], label="Training Loss")
            ax[0].plot(epochcount, [v.detach().numpy() for v in testloss], label="Test Loss")
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

            match1, match2, match3 = torch.split(X_test, 1, dim=0)
            test_match1 = encoder.inverse_transform(match1.reshape(2, 3)).reshape(2)
            test_match2 = encoder.inverse_transform(match2.reshape(2, 3)).reshape(2)
            test_match3 = encoder.inverse_transform(match3.reshape(2, 3)).reshape(2)

            test_match1_string = f"{test_match1[0]} vs {test_match1[1]}"
            test_match2_string = f"{test_match2[0]} vs {test_match2[1]}"
            test_match3_string = f"{test_match3[0]} vs {test_match3[1]}"

            test_graph_results = torch.sigmoid(test_logits).transpose(0,1).squeeze()

            ax[2].bar([test_match1_string, test_match2_string, test_match3_string], test_graph_results)
            ax[2].set_title("Test Matches And Predictions After Training")
            ax[2].set_ylim(bottom=0, top=1)
            ax[2].set_ylabel("Chance that Team 1 wins")
            ax[2].set_xlabel("Test Matches")

            plt.show()