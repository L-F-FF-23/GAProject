import torch
from sklearn.model_selection import train_test_split
from torch import nn
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np

encoder = LabelEncoder()

with open("data.json", "r") as f:
    data = json.load(f)

allteams = []

for match in data:
    allteams.append(data[match]["Team1"])
    allteams.append(data[match]["Team2"])

finishedteams = list(set(allteams))
fulldatatensor = torch.zeros(len(data), 3)

encoder.fit(allteams)


loopnum = 0

for match in data:

    x = encoder.transform([data[match]["Team1"]])
    y = encoder.transform([data[match]["Team2"]])
    z = encoder.transform([data[match]["Winner"]])

    numpyarray = np.array([x, y, z], dtype=np.float32)
    matchtensorunflipped = torch.from_numpy(numpyarray).float()
    matchtensor = torch.transpose(matchtensorunflipped,0, 1)

    fulldatatensor[loopnum] = matchtensor
    loopnum += 1

X_data, y_data = torch.split(fulldatatensor, [2,1],1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

y_train, y_test = y_train.long().squeeze(), y_test.long().squeeze()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.passthrough = nn.Sequential(
            nn.Linear(2, 5),
            nn.Linear(5, 5),
            nn.Linear(5, len(encoder.classes_)),
        )

    def forward(self, x):
        return self.passthrough(x)

model1 = Net()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()

    accuracy = correct / len(y_pred) * 100
    return accuracy

epochs = 10000

for epoch in range(epochs):
    model1.train()

    y_logits = model1(X_train)

    y_preds = torch.argmax(y_logits, dim=1)

    loss = loss_fn(y_logits, y_train)

    acc = accuracy_fn(y_true=y_train, y_pred=y_preds)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    model1.eval()
    with torch.inference_mode():

        test_logits = model1(X_test)
        test_preds = torch.argmax(test_logits, dim=1)

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)

        if epoch == 99:
            print(test_loss)
            print(test_acc)
            print(encoder.inverse_transform(test_preds))
            test_Match1 = X_test[0,:].int()
            test_Match2 = X_test[1,:].int()
            test_Match3 = X_test[2,:].int()
            print(encoder.inverse_transform(test_Match1), encoder.inverse_transform(test_Match2), encoder.inverse_transform(test_Match3))