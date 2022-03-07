import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from dataset import get_datasets
#from data_loader_detection import DatasetPoison -- use the dataset loader from hypernets code
import argparse

import warnings

warnings.filterwarnings("ignore")

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--lmd', default=0.1)
parser.add_argument('--gamma', default=0.1)
parser.add_argument('--seed', default=8)
parser.add_argument('--lr', default=0.0003)
args = parser.parse_args()

# Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 11
hidden_size = 110
num_classes = 11
num_epochs = 6000
batch_size = 50
learning_rate = float(args.lr)
meta_batch = 3
lmd = float(args.lmd)
gamma = float(args.gamma)
seed = int(args.seed)

train_loader, val_loader, test_loader = get_datasets('cifar10', 'data', normalize=True, val_size=10000)
print(train_loader)
#train_loader, test_loader = DatasetPoison().poisoned_loader_shift(batch_size=batch_size)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class MLP_LR_context_detection_regularization(nn.Module):
    def __init__(self, input_size, hidden_size, vector_size):
        super(MLP_LR_context_detection_regularization, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.context = nn.Linear(hidden_size, vector_size)
        self.LR_prediction = nn.Linear(input_size + vector_size, 1)
        self.LR_detection = nn.Linear(input_size + vector_size, 1)
        self.LR_context = nn.Linear(vector_size, 1)
        self.vector_size = vector_size

    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu2(hidden2)

        context_vector = self.context(relu2) #yes
        context_prediction_vectors = self.context(relu2)

        ###### adaptive prediction - we need to add this to our code to concat the context vector with the data
        prediction_vector = torch.mean(context_vector, dim=0)
        prediction_vector = prediction_vector.expand(len(x), self.vector_size)
        prediction_vector_x = torch.cat((prediction_vector, x), dim=1)
        y_prediction = self.LR_prediction(prediction_vector_x)
        y_prediction = torch.sigmoid(y_prediction)

        ###### context learning
        y_context_prediction = self.LR_context(context_prediction_vectors)
        y_context_prediction = torch.sigmoid(y_context_prediction)

        ###### adaptive detection
        detection_vector = torch.mean(context_vector, dim=0)
        detection_vector = detection_vector.expand(len(x), self.vector_size)
        detection_vector_x = torch.cat((detection_vector, x), dim=1)
        y_detection = self.LR_detection(detection_vector_x)
        y_detection = torch.sigmoid(y_detection)

        return y_prediction, y_detection, y_context_prediction


model = MLP_LR_context_detection_regularization(input_size, hidden_size, vector_size=11).to(device)

# Loss and optimizer
criterion_prediction = torch.nn.BCELoss()
criterion_detection = torch.nn.BCELoss()
criterion_context = torch.nn.KLDivLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
train_loss = []
test_loss = []
train_acc_prediction = []
train_acc_detection = []
test_acc_prediction = []
test_acc_detection = []

train_f1_prediction = []
train_f1_detection = []
test_f1_prediction = []
test_f1_detection = []

train_recall_prediction = []
train_recall_detection = []
test_recall_prediction = []
test_recall_detection = []

train_acc_group_detc = [[] for i in range(10)]
test_acc_group_detc = [[] for i in range(7)]

break_for = False

output = "Setting: lambda = {}, gamma = {}, seed = {} \n".format(lmd, gamma, seed)

for epoch in range(num_epochs):
    group_ids = random.sample(range(0, len(train_loader)), meta_batch)
    loss = 0
    total_loss_prediction = 0
    total_loss_detection = 0

    for index in group_ids:
        images, labels = next(iter(train_loader[index]))
        images = images.to(device)
        labels_pred = labels[:, 0].to(device)
        labels_dec = labels[:, 1].to(device)
        # Forward pass
        y_pre, y_dec, y_context = model(images)
        loss_prediction = criterion_prediction(y_pre, torch.reshape(labels_pred, (-1, 1)))
        loss_detection = criterion_detection(y_dec, torch.reshape(labels_dec, (-1, 1)))
        loss_context = criterion_context(torch.cat((y_pre, 1 - y_pre), dim=1),
                                         torch.cat((y_context, 1 - y_context), dim=1))
        loss += loss_prediction + lmd * loss_detection + gamma * torch.abs(loss_prediction - loss_context)

    loss = loss / meta_batch
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc = 0

    if (epoch + 1) % 50 == 0:
        if epoch == num_epochs - 1:
            print('Epoch [{}/{}], Total Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, loss.item()))

        with torch.no_grad():
            correct_prediction = 0
            correct_detection = 0
            total = 0

            TP_prediction = 0
            FP_prediction = 0
            FN_prediction = 0

            TP_detection = 0
            FP_detection = 0
            FN_detection = 0

            for i in range(len(train_loader)):

                group_total = 0
                group_correct_detection = 0

                for images, labels in train_loader[i]:
                    images = images.to(device)
                    labels_pred = labels[:, 0].to(device)
                    labels_dec = labels[:, 1].to(device)
                    y_pre, y_dec, y_context = model(images)

                    predicted_prediction = y_pre.round().reshape(1, len(y_pre))
                    predicted_detection = y_dec.round().reshape(1, len(y_dec))
                    total += labels_pred.size(0)
                    correct_prediction += (predicted_prediction.eq(labels_pred)).sum().item()
                    correct_detection += (predicted_detection.eq(labels_dec)).sum().item()

                    group_total += labels_pred.size(0)
                    group_correct_detection += (predicted_detection.eq(labels_dec)).sum().item()

                    predicted_prediction = predicted_prediction.type(torch.IntTensor).numpy().reshape(-1)
                    predicted_detection = predicted_detection.type(torch.IntTensor).numpy().reshape(-1)
                    labels_pred = labels_pred.type(torch.IntTensor).numpy().reshape(-1)
                    labels_dec = labels_dec.type(torch.IntTensor).numpy().reshape(-1)

                    TP_prediction += np.count_nonzero(predicted_prediction & labels_pred)
                    TP_detection += np.count_nonzero(predicted_detection & labels_dec)
                    FP_prediction += np.count_nonzero((predicted_prediction == 0) & (labels_pred == 1))
                    FP_detection += np.count_nonzero((predicted_detection == 0) & (labels_dec == 1))
                    FN_prediction += np.count_nonzero((predicted_prediction == 1) & (labels_pred == 0))
                    FN_detection += np.count_nonzero((predicted_detection == 1) & (labels_dec == 0))

                train_acc_group_detc[i].append(100 * group_correct_detection * 1.0 / group_total)

            train_loss.append(loss.item())

            f1_score_prediction = TP_prediction / (TP_prediction + (FP_prediction + FN_prediction) / 2)
            f1_score_detection = TP_detection / (TP_detection + (FP_detection + FN_detection) / 2)
            if epoch == num_epochs - 1:
                print('Accuracy of the training example - Prediction: {} %'.format(
                    100 * correct_prediction * 1.0 / total))
                print('F1 score of the training example - Prediction: {}'.format(f1_score_prediction))
                # print('F1 score of the training example - Prediction: {} %'.format(
                #     f1_score(np.array(actl_pred), np.array(pred_pred))))
                print(
                    'Accuracy of the training example - Detection: {} %'.format(100 * correct_detection * 1.0 / total))
                print('F1 score of the training example - Detection: {}'.format(f1_score_detection))
                # print('F1 score of the training example - Detection: {} %'.format(
                #     f1_score(np.array(actl_detc), np.array(pred_detc))))

                output += 'Accuracy of the training example - Prediction: {} % \n'.format(
                    100 * correct_prediction * 1.0 / total)
                output += 'F1 score of the training example - Prediction: {} \n'.format(f1_score_prediction)

                output += 'Accuracy of the training example - Detection: {} % \n'.format(
                    100 * correct_detection * 1.0 / total)
                output += 'F1 score of the training example - Detection: {} \n'.format(f1_score_detection)

            train_acc_prediction.append(100 * correct_prediction * 1.0 / total)
            train_acc_detection.append(100 * correct_detection * 1.0 / total)

            train_f1_prediction.append(f1_score_prediction)
            train_f1_detection.append(f1_score_detection)

            train_acc = 100 * correct_prediction * 1.0 / total
        #
        with torch.no_grad():
            correct_prediction = 0
            correct_detection = 0
            total = 0
            total_loss = 0
            number_of_points = 0

            TP_prediction = 0
            FP_prediction = 0
            FN_prediction = 0

            TP_detection = 0
            FP_detection = 0
            FN_detection = 0

            for i in range(len(test_loader)):
                group_correct_detection = 0
                group_total = 0
                for images, labels in test_loader[i]:
                    images = images.to(device)
                    labels_pred = labels[:, 0].to(device)
                    labels_dec = labels[:, 1].to(device)
                    y_pre, y_dec, y_context = model(images)

                    predicted_prediction = y_pre.round().reshape(1, len(y_pre))
                    predicted_detection = y_dec.round().reshape(1, len(y_dec))

                    group_total += labels_pred.size(0)
                    # group_correct_prediction += (predicted_prediction.eq(labels_pred)).sum().item()
                    group_correct_detection += (predicted_detection.eq(labels_dec)).sum().item()

                    total += labels_pred.size(0)
                    correct_prediction += (predicted_prediction.eq(labels_pred)).sum().item()
                    correct_detection += (predicted_detection.eq(labels_dec)).sum().item()

                    loss_prediction = criterion_prediction(y_pre, torch.reshape(labels_pred, (-1, 1)))
                    loss_detection = criterion_detection(y_dec, torch.reshape(labels_dec, (-1, 1)))
                    total_loss += loss_prediction.item() + loss_detection.item()  # two losses
                    number_of_points += 1

                    predicted_prediction = predicted_prediction.type(torch.IntTensor).numpy().reshape(-1)
                    predicted_detection = predicted_detection.type(torch.IntTensor).numpy().reshape(-1)
                    labels_pred = labels_pred.type(torch.IntTensor).numpy().reshape(-1)
                    labels_dec = labels_dec.type(torch.IntTensor).numpy().reshape(-1)

                    TP_prediction += np.count_nonzero(predicted_prediction & labels_pred)
                    TP_detection += np.count_nonzero(predicted_detection & labels_dec)
                    FP_prediction += np.count_nonzero((predicted_prediction == 0) & (labels_pred == 1))
                    FP_detection += np.count_nonzero((predicted_detection == 0) & (labels_dec == 1))
                    FN_prediction += np.count_nonzero((predicted_prediction == 1) & (labels_pred == 0))
                    FN_detection += np.count_nonzero((predicted_detection == 1) & (labels_dec == 0))


                test_acc_group_detc[i].append(100 * group_correct_detection * 1.0 / group_total)

            f1_score_prediction = TP_prediction / (TP_prediction + (FP_prediction + FN_prediction) / 2)
            f1_score_detection = TP_detection / (TP_detection + (FP_detection + FN_detection) / 2)

            if epoch == num_epochs - 1:
                print('Accuracy of the test example - Prediction: {} %'.format(100 * correct_prediction * 1.0 / total))
                print('F1 score of the test example - Prediction: {}'.format(f1_score_prediction))
                print('Accuracy of the test example - Detection: {} %'.format(100 * correct_detection * 1.0 / total))
                print('F1 score of the test example - Detection: {}'.format(f1_score_detection))

                output += 'Accuracy of the test example - Prediction: {} % \n'.format(
                    100 * correct_prediction * 1.0 / total)
                output += 'F1 score of the test example - Prediction: {} \n'.format(f1_score_prediction)

                output += 'Accuracy of the test example - Detection: {} % \n'.format(
                    100 * correct_detection * 1.0 / total)
                output += 'F1 score of the test example - Detection: {} \n'.format(f1_score_detection)

            test_loss.append(total_loss / number_of_points)
            test_acc_prediction.append(100 * correct_prediction * 1.0 / total)
            test_acc_detection.append(100 * correct_detection * 1.0 / total)

            test_f1_prediction.append(f1_score_prediction)
            test_f1_detection.append(f1_score_detection)

# Group test accuracy
# with torch.no_grad():
#     correct_prediction = 0
#     correct_detection = 0
#     total = 0
#     total_loss = 0
#     number_of_points = 0
#
#     TP_prediction = 0
#     FP_prediction = 0
#     FN_prediction = 0
#
#     TP_detection = 0
#     FP_detection = 0
#     FN_detection = 0
#
#     correct_prediction_clean = 0
#     correct_detection_clean = 0
#     correct_prediction_attack = [0, 0, 0, 0]
#     correct_detection_attack = [0, 0, 0, 0]
#     attack_idx = 0
#
#     for i in range(len(test_loader)):
#         for images, labels in test_loader[i]:
#             images = images.to(device)
#             labels_pred = labels[:, 0].to(device)
#             labels_dec = labels[:, 1].to(device)
#             y_pre, y_dec, y_context = model(images)
#
#             predicted_prediction = y_pre.round().reshape(1, len(y_pre))
#             predicted_detection = y_dec.round().reshape(1, len(y_dec))
#             total += labels_pred.size(0)
#             correct_prediction += (predicted_prediction[0].eq(labels_pred)).sum().item()
#             correct_detection += (predicted_detection[0].eq(labels_dec)).sum().item()
#
#             indices = (labels_dec == 1).nonzero(as_tuple=True)[0]
#             if len(indices) > 0:
#                 correct_prediction_attack[int(attack_idx)] += (
#                     predicted_prediction[0].eq(labels_pred)).sum().item()
#                 correct_detection_attack[int(attack_idx)] += (
#                     predicted_detection[0].eq(labels_dec)).sum().item()
#                 attack_idx += 0.5
#             else:
#                 correct_prediction_clean += (predicted_prediction[0].eq(labels_pred)).sum().item()
#                 correct_detection_clean += (predicted_detection[0].eq(labels_dec)).sum().item()
#
#     print('Final all - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction * 1.0 / total))
#
#     print('Final all - Accuracy of the test example - Detection: {} %'.format(
#         100 * correct_detection * 1.0 / total))
#
#     print('Final test clean - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_clean * 1.0 / 300))
#     print('Final test clean - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_detection_clean * 1.0 / 300))
#
#     print('Final attack 1 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_attack[0] * 1.0 / 100))
#     print('Final attack 1 - Accuracy of the test example - Detection: {} %'.format(
#         100 * correct_detection_attack[0] * 1.0 / 100))
#     print('Final attack 2 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_attack[1] * 1.0 / 100))
#     print('Final attack 2 - Accuracy of the test example - Detection: {} %'.format(
#         100 * correct_detection_attack[1] * 1.0 / 100))
#     print('Final attack 3 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_attack[2] * 1.0 / 100))
#     print('Final attack 3 - Accuracy of the test example - Detection: {} %'.format(
#         100 * correct_detection_attack[2] * 1.0 / 100))
#     print('Final attack 4 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_attack[3] * 1.0 / 100))
#     print('Final attack 4 - Accuracy of the test example - Detection: {} %'.format(
#         100 * correct_detection_attack[3] * 1.0 / 100))


# with torch.no_grad():
#     correct_prediction = 0
#     correct_detection = 0
#     total = 0
#     total_loss = 0
#     number_of_points = 0
#
#     TP_prediction = 0
#     FP_prediction = 0
#     FN_prediction = 0
#
#     TP_detection = 0
#     FP_detection = 0
#     FN_detection = 0
#
#     correct_prediction_clean = 0
#     correct_detection_clean = 0
#     correct_prediction_attack = [0, 0, 0, 0]
#     correct_detection_attack = [0, 0, 0, 0]
#     attack_idx = 0
#
#     for i in range(len(test_loader)):
#         for images, labels in test_loader[i]:
#             images = images.to(device)
#             labels_pred = labels[:, 0].to(device)
#             labels_dec = labels[:, 1].to(device)
#             y_pre, y_dec, y_context = model(images)
#
#             predicted_prediction = y_pre.round().reshape(1, len(y_pre))
#             predicted_detection = y_dec.round().reshape(1, len(y_dec))
#             total += labels_pred.size(0)
#             correct_prediction += (predicted_prediction[0].eq(labels_pred)).sum().item()
#             correct_detection += (predicted_detection[0].eq(labels_dec)).sum().item()
#
#             indices = (labels_dec == 0).nonzero(as_tuple=True)[0]
#             correct_prediction_clean += (predicted_prediction[0][indices].eq(labels_pred[indices])).sum().item()
#             correct_detection_clean += (predicted_detection[0][indices].eq(labels_dec[indices])).sum().item()
#
#             indices = (labels_dec == 1).nonzero(as_tuple=True)[0]
#             if len(indices) > 0:
#                 correct_prediction_attack[int(attack_idx)] += (
#                     predicted_prediction[0][indices].eq(labels_pred[indices])).sum().item()
#                 correct_detection_attack[int(attack_idx)] += (
#                     predicted_detection[0][indices].eq(labels_dec[indices])).sum().item()
#                 attack_idx += 0.5
#
#     print('Final all - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction * 1.0 / total))
#
#     print('Final all - Accuracy of the test example - Detection: {} %'.format(
#         100 * correct_detection * 1.0 / total))
#
#     print('Final test clean - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_clean * 1.0 / 500))
#     print('Final test clean - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_detection_clean * 1.0 / 500))
#
#     print('Final attack 1 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_attack[0] * 1.0 / 50))
#     print('Final attack 1 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_detection_attack[0] * 1.0 / 50))
#     print('Final attack 2 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_attack[1] * 1.0 / 50))
#     print('Final attack 2 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_detection_attack[1] * 1.0 / 50))
#     print('Final attack 3 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_attack[2] * 1.0 / 50))
#     print('Final attack 3 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_detection_attack[2] * 1.0 / 50))
#     print('Final attack 4 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_attack[3] * 1.0 / 50))
#     print('Final attack 4 - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_detection_attack[3] * 1.0 / 50))

# with torch.no_grad():
#     correct_prediction = 0
#     correct_detection = 0
#     total = 0
#     total_loss = 0
#     number_of_points = 0
#
#     TP_prediction = 0
#     FP_prediction = 0
#     FN_prediction = 0
#
#     TP_detection = 0
#     FP_detection = 0
#     FN_detection = 0
#
#     correct_prediction_clean = 0
#     correct_detection_clean = 0
#     correct_prediction_attack = [0, 0, 0, 0,0,0,0,0,0]
#     correct_detection_attack = [0, 0, 0, 0,0,0,0,0,0]
#     attack_idx = 0
#
#     for i in range(len(train_loader)):
#         print(">>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<")
#         for images, labels in train_loader[i]:
#             images = images.to(device)
#             labels_pred = labels[:, 0].to(device)
#             labels_dec = labels[:, 1].to(device)
#             y_pre, y_dec, y_context = model(images)
#
#             predicted_prediction = y_pre.round().reshape(1, len(y_pre))
#             predicted_detection = y_dec.round().reshape(1, len(y_dec))
#             total += labels_pred.size(0)
#             correct_prediction += (predicted_prediction[0].eq(labels_pred)).sum().item()
#             correct_detection += (predicted_detection[0].eq(labels_dec)).sum().item()
#
#             indices = (labels_dec == 0).nonzero(as_tuple=True)[0]
#             correct_prediction_clean += (predicted_prediction[0][indices].eq(labels_pred[indices])).sum().item()
#             correct_detection_clean += (predicted_detection[0][indices].eq(labels_dec[indices])).sum().item()
#
#             indices = (labels_dec == 1).nonzero(as_tuple=True)[0]
#             if len(indices) > 0:
#                 print(len(indices))
#                 correct_prediction_attack[int(attack_idx)] += (
#                     predicted_prediction[0][indices].eq(labels_pred[indices])).sum().item()
#                 correct_detection_attack[int(attack_idx)] += (
#                     predicted_detection[0][indices].eq(labels_dec[indices])).sum().item()
#                 attack_idx += 0.125
#
#     print('Final all - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction * 1.0 / total))
#
#     print('Final all - Accuracy of the test example - Detection: {} %'.format(
#         100 * correct_detection * 1.0 / total))
#
#     print('Final test clean - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_prediction_clean * 1.0 / 2200))
#     print('Final test clean - Accuracy of the test example - Prediction: {} %'.format(
#         100 * correct_detection_clean * 1.0 / 2200))
#
#     for i in range(9):
#         print('Final attack {} - Accuracy of the test example - Prediction: {} %'.format(i+1,
#             100 * correct_prediction_attack[i] * 1.0 / 200))
#         print('Final attack {} - Accuracy of the test example - Prediction: {} %'.format(i+1,
#             100 * correct_detection_attack[i] * 1.0 / 200))

# Save the model checkpoint

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(train_f1_prediction, color='b', label='Pred')
# axs[0, 0].plot(train_f1_detection, color='r', label='Detc')
# axs[0, 0].set_title('Train F1')
#
# axs[0, 1].plot(test_f1_prediction, color='b', label='Pred')
# axs[0, 1].plot(test_f1_detection, color='r', label='Detc')
# axs[0, 1].set_title('Test F1')
#
# axs[1, 0].plot(train_acc_prediction, color='b', label='Pred')
# axs[1, 0].plot(train_acc_detection, color='r', label='Detc')
# axs[1, 0].set_title('Train acc.')
#
# axs[1, 1].plot(test_acc_prediction, color='b', label='Pred')
# axs[1, 1].plot(test_acc_detection, color='r', label='Detc')
# axs[1, 1].set_title('Test acc.')
#
# # for ax in axs.flat:
# axs.flat[0].set(ylabel='f1 score')
# axs.flat[2].set(xlabel='iters', ylabel='acc')
# axs.flat[3].set(xlabel='iters')

# for ax in axs.flat:
#     ax.label_outer()

# plt.legend()
# plt.show()

# plt.plot(train_f1_prediction, color='b', label='Pred')
# plt.plot(train_f1_detection, color='r', label='Detc')
# plt.legend()
# plt.ylabel('f1 score')
# plt.xlabel('iters')
# plt.show()
# plt.plot(test_f1_prediction, color='b', label='Pred')
# plt.plot(test_f1_detection, color='r', label='Detc')
# plt.ylabel('f1 score')
# plt.xlabel('iters')
# plt.legend()
# plt.show()
# plt.plot(train_acc_prediction, color='b', label='Pred')
# plt.plot(train_acc_detection, color='r', label='Detc')
# plt.legend()
# plt.ylabel('acc')
# plt.xlabel('iters')
# plt.show()
# plt.plot(test_acc_prediction, color='b', label='Pred')
# plt.plot(test_acc_detection, color='r', label='Detc')
# plt.ylabel('acc')
# plt.xlabel('iters')
# plt.legend()
# plt.show()
# plt.plot(train_recall_prediction, color='b', label='Pred')
# plt.plot(train_recall_detection, color='r', label='Detc')
# plt.legend()
# plt.ylabel('recall')
# plt.xlabel('iters')
# plt.show()
# plt.plot(test_recall_prediction, color='b', label='Pred')
# plt.plot(test_recall_detection, color='r', label='Detc')
# plt.ylabel('recall')
# plt.xlabel('iters')
# plt.legend()
# plt.show()

# plt.plot(train_acc_group_detc[0], color='b', label='Group 0')
# plt.plot(train_acc_group_detc[1], color='r', label='Group 1')
# plt.plot(train_acc_group_detc[2], color='y', label='Group 2')
# plt.plot(train_acc_group_detc[3], color='orange', label='Group 3')
# plt.plot(train_acc_group_detc[4], color='g', label='Group 4')
# plt.plot(train_acc_group_detc[5], label='Group 5')
# plt.plot(train_acc_group_detc[6], label='Group 6')
# plt.plot(train_acc_group_detc[7], label='Group 7')
# plt.plot(train_acc_group_detc[8], label='Group 8')
# plt.plot(train_acc_group_detc[9], label='Group 9')
# plt.ylabel('accuracy')
# plt.xlabel('iters')
# plt.legend()
# plt.show()
#
# plt.plot(test_acc_group_detc[0], color='b', label='Group 0')
# plt.plot(test_acc_group_detc[1], color='r', label='Group 1')
# plt.plot(test_acc_group_detc[2], color='y', label='Group 2')
# plt.plot(test_acc_group_detc[3], color='orange', label='Group 3')
# plt.plot(test_acc_group_detc[4], color='g', label='Group 4')
# plt.plot(test_acc_group_detc[5], label='Group 5')
# plt.plot(test_acc_group_detc[6], label='Group 6')
# plt.ylabel('accuracy')
# plt.xlabel('iters')
# plt.legend()
# plt.show()
#
# plt.plot(train_loss, color='b', label='Train')
# plt.plot(test_loss, color='r', label='Test')
# plt.ylabel('loss')
# plt.xlabel('iters')
# plt.legend()
# plt.show()
torch.save(model.state_dict(), 'model.ckpt')

with open("AAD_reg_results_group.txt", 'a') as f:
    f.writelines(output)
