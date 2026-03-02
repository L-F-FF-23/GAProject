import torch
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

torch.manual_seed(42) # sets a random seed, which is used for controlled randomness

fig, ax = plt.subplots(1, 3, figsize=(15, 5)) # base plot which then the 3 graph in the end gets added to

# two lists which will include every single team and champion that the code finds in the csv, has duplicates
allteams = []
allchamps = []

# opens the csv file and saves it as a list so the rest of the code can access the data
with open(file="2025_LoL_esports_match_data_from_OraclesElixir.csv", mode="r", newline="", encoding="utf-8") as f:
    data = csv.DictReader(f)
    datalist = list(data)


h2h = defaultdict(lambda: [0, 0]) # head 2 head, it's a defaultdict which creates a list with 2 zeroes whenever called if the h2h[data] doesn't exist

# looks through the matches in the data and takes every teams name and the 5 champions both teams played
for row in datalist:
    if row["participantid"] == "100" or row["participantid"] == "200":
        allteams.append(row["teamname"])
        allchamps.append(row["pick1"])
        allchamps.append(row["pick2"])
        allchamps.append(row["pick3"])
        allchamps.append(row["pick4"])
        allchamps.append(row["pick5"])

matchCount = len(allteams)//2

# takes the full lists and removes all duplicates
finishedteams = np.unique(np.array(allteams)).reshape(-1)
finishedchamps = np.unique(np.array(allchamps)).reshape(-1)

# empty dicts which then gets filled with every team with an assigned number and every champion with an assigned number (and vice versa)
teamsnametoids = {}
teamsidtonames = {}
champsnametoids = {}
champsidtonames = {}
for idx, name in enumerate(finishedteams):
    teamsnametoids[name] = idx
    teamsidtonames[idx] = name
for idx, name in enumerate(finishedchamps):
    champsnametoids[name] = idx
    champsidtonames[idx] = name

twoteamfeats = 2 + 2 + 10 # used to know how many feats a full match has before embedding

# creates two empty tensors which will be filled with the file data later on
teamsdata = torch.zeros(matchCount, twoteamfeats)
resultsdata = torch.zeros(matchCount, 1)

loopnum = 0

# sets default values, this only exists to prevent the code complaining at me for a value being able to be unassigned
T1 = False
T2 = False
x = None
y = None
z = None
team1 = None
team2 = None
champ1 = None
champ2 = None
champ3 = None
champ4 = None
champ5 = None
champ6 = None
champ7 = None
champ8 = None
champ9 = None
champ10 = None

# goes through every match and takes the teams names, the champions played and makes a winrate counter (if one already exists it will instead add onto the existing one)
for row in datalist:
    if row["participantid"] == "100":
        x = teamsnametoids[row["teamname"]]
        champ1 = champsnametoids[row["pick1"]]
        champ2 = champsnametoids[row["pick2"]]
        champ3 = champsnametoids[row["pick3"]]
        champ4 = champsnametoids[row["pick4"]]
        champ5 = champsnametoids[row["pick5"]]
        team1 = row["teamname"]
        if row["result"] == "1":
            z = 1
        else:
            z = 0
        T1 = True

    if row["participantid"] == "200":
        y = teamsnametoids[row["teamname"]]
        champ6 = champsnametoids[row["pick1"]]
        champ7 = champsnametoids[row["pick2"]]
        champ8 = champsnametoids[row["pick3"]]
        champ9 = champsnametoids[row["pick4"]]
        champ10 = champsnametoids[row["pick5"]]
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
        numpyarray = np.concatenate((x, champ1, champ2, champ3, champ4, champ5, y, champ6, champ7, champ8, champ9, champ10, winrates), axis=None, dtype=np.float32)
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


# finished datasets
X_data = teamsdata
y_data = resultsdata

# splits into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True)

# makes them into datasets which is useful for batching and shuffling
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# the models code
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # embedding, it's a way for the model to turn numbers into a separate tensor to more easily recognize it when computing, so every team and champions name gets turned into a 1d tensors with 8 random numbers each
        self.team_embedding = nn.Embedding(num_embeddings=len(finishedteams), embedding_dim=8)
        self.champ_embedding = nn.Embedding(num_embeddings=len(finishedchamps), embedding_dim=8)

        # This is the sequence that the model does after all the data has been properly transformed
        self.passthrough = nn.Sequential(
            nn.Linear(98, 100), # applies simple linear y = kx + m
            nn.ReLU(), # allows the model to do non-linearity, it does this through math I don't fully understand
            nn.Dropout(0.1), # disables an amount of neurons in the layer, this way the neural network can't rely on certain neurons and needs to find patterns
            nn.Linear(100, 75),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, 1) # the last out feature here is the probability of team 1 winning in logits
        )

    def forward(self, x):
        modelteams, modelchamps, modelwr = torch.cat((x[:, 0], x[:, 6]), dim=0), torch.cat((x[:, 1:6], x[:, 7:12]), dim=0), x[:, -2:] # splits up the input batch for embedding

        # both needs to be a long otherwise embedding breaks
        modelteams = modelteams.long()
        modelchamps = modelchamps.long()

        # here the model takes in all the teams in one tensor and the champions in another, and turns them from a digit to multiple ones. This is to prevent the model from thinking one team is stronger than another just because one team has the digit 4 and another is 103 for example
        teamsemb = self.team_embedding(modelteams)
        champsemb = self.champ_embedding(modelchamps)

        # just reorients the tensor to be able to be remerged into the data
        teamsemb = teamsemb.view(-1, 16)
        champsemb = champsemb.view(-1, 80)
        x = torch.cat((teamsemb, champsemb, modelwr), dim=1) # puts back all the tensors into one so it can be processed by the model
        return self.passthrough(x)

model1 = Net() # puts the model in a usable variable

loss_fn = nn.BCEWithLogitsLoss() # loss function used to calculate how far off the model is, uses WithLogitsLoss to avoid coding that in automatically

optimizer = torch.optim.AdamW(model1.parameters(), lr=0.0002, weight_decay=0.0005) # optimizer, lr (learning rate) = how big steps the model takes when adjusting, weight_decay = penalizes the model for updating the weights too much to prevent overfitting

def accuracy_fn(y_true, y_pred): # simple function to calculate the accuracy of the model
    correct = torch.eq(y_true, y_pred).sum().item()

    accuracy = correct / len(y_pred) * 100
    return accuracy

epochs = 500 # how many training and test loops
trainloss = []
testloss = []
trainaccuracy = []
testaccuracy = []
epochcount = []

for epoch in range(epochs):
    model1.train() # sets the model in training mode to allow certain layers to act a certain way during the training of the model
    train_batch_preds = []
    train_batch_labels = []
    train_batch_loss = 0

    for X_batch, y_batch in train_loader:
        y_logits = model1(X_batch) # sends in a batch of tensors and returns the logit result
        loss = loss_fn(y_logits, y_batch) # put the models result and the real result into the loss function and gives out the loss
        optimizer.zero_grad() # resets optimizer between steps to prevent the optimizer for accumulating
        loss.backward() # the model looks at the loss and compares it with its own parameters
        optimizer.step() # the model then uses the optimizer to change the bias and weight of the parameters to improve loss

        train_batch_loss += loss.item() * len(y_batch) # multiply the loss with batch size to give all the losses the same weight when putting it together
        train_batch_preds.append(torch.sigmoid(y_logits)) # sigmoid is used here to turn logits into actual chance decimals, this is what using WithLogitsLoss bypasses when feeding into the loss function
        train_batch_labels.append(y_batch)

    train_epoch_preds = torch.cat(train_batch_preds)
    train_epoch_labels = torch.cat(train_batch_labels) # simply takes all the batches predictions and labels and turns them into one tensor for the epoch
    train_epoch_loss = train_batch_loss / len(train_loader.dataset) # need to divide by train_loader.dataset size to account for different batch sizes
    train_epoch_acc = accuracy_fn(y_true=train_epoch_labels, y_pred=torch.round(train_epoch_preds)) # runs the accuracy function to check how many times the model guessed in favour of the team that actually won

    trainloss.append(train_epoch_loss)
    trainaccuracy.append(train_epoch_acc)

    model1.eval() # evaluation mode used for testing and changes behavior in certain layers
    with (torch.inference_mode()): # disables gradient decent among other things needed for the testing process
        # then it's basically just a repeat of the training function without the optimization, testing is only used so the creator can view progress on unseen data
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

        # creates the graphs for training and testing, loss and accuracy
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

            # this batch of code takes 3 of the testing matchups that were ran in the final epoch and plots them into the matplot window, then it executes plt.show() to show all the graphs plotted throughout the code
            test_match1_team1, test_match1_team2  = X_test[0,0].int().item(), X_test[0,6].int().item()
            test_match2_team1, test_match2_team2  = X_test[1,0].int().item(), X_test[1,6].int().item()
            test_match3_team1, test_match3_team2  = X_test[2,0].int().item(), X_test[2,6].int().item()

            test_match1_string = f"{teamsidtonames[test_match1_team1]} vs {teamsidtonames[test_match1_team2]}"
            test_match2_string = f"{teamsidtonames[test_match2_team1]} vs {teamsidtonames[test_match2_team2]}"
            test_match3_string = f"{teamsidtonames[test_match3_team1]} vs {teamsidtonames[test_match3_team2]}"

            test_graph_results = torch.sigmoid(torch.cat((test_logits[0], test_logits[1], test_logits[2]))) #.transpose(0,1).squeeze()

            ax[2].bar([test_match1_string, test_match2_string, test_match3_string], test_graph_results)
            ax[2].set_title("Test Matches And Predictions After Training")
            ax[2].set_ylim(bottom=0, top=1)
            ax[2].set_ylabel("Chance that Team 1 wins")
            ax[2].set_xlabel("Test Matches")

            plt.show()

# finally this saves the parameters of the model so if another person had this code they could simply import the parameters (from src/models) without needing to train it
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_Saved_Net_Model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model1.state_dict(), f=MODEL_SAVE_PATH)