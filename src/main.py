import torch
from sklearn.model_selection import train_test_split
from torch import nn
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np

torch.manual_seed(42)

encoder = OneHotEncoder()

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

epochs = 1000

for epoch in range(epochs):
    model1.train()

    y_logits = model1(X_train)

    loss = loss_fn(y_logits, y_train)

    print(f"Train Loss: {loss}")

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model1.eval()
    with torch.inference_mode():

        test_logits = model1(X_test)
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)

        print(f"Test Loss: {test_loss}")

        if epoch == 999:
            print(test_acc)
            print((torch.sigmoid(test_logits)).transpose(0,1).squeeze())
            match1, match2, match3 = torch.split(X_test, 1, dim=0)
            test_Match1 = match1.reshape(2, 3)
            test_Match2 = match2.reshape(2, 3)
            test_Match3 = match3.reshape(2, 3)
            print(encoder.inverse_transform(test_Match1).reshape(2))
            print(encoder.inverse_transform(test_Match2).reshape(2))
            print(encoder.inverse_transform(test_Match3).reshape(2))