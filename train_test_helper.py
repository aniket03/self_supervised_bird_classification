import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_


def get_count_correct_preds(network_output, target):

    output = network_output
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    pred.data = pred.data.view_as(target.data)
    correct = target.eq(pred).sum().item()

    return correct


class ModelTrainTest():

    def __init__(self, network, device, model_file_path, threshold=1e-4):
        super(ModelTrainTest, self).__init__()
        self.network = network
        self.device = device
        self.model_file_path = model_file_path
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9

    def train(self, optimizer, epoch, params_max_norm, train_data_loader, val_data_loader):
        self.network.train()
        train_loss = 0
        correct = 0
        cnt_batches = 0

        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = Variable(data).to(self.device), Variable(target).to(self.device)

            optimizer.zero_grad()
            output = self.network(data)

            loss = F.nll_loss(output, target)
            loss.backward()

            clip_grad_norm_(self.network.parameters(), params_max_norm)
            optimizer.step()

            correct += get_count_correct_preds(output, target)
            train_loss += loss.item()
            cnt_batches += 1

            del data, target, output

        train_loss /= cnt_batches
        val_loss, val_acc = self.test(epoch, val_data_loader)

        if val_loss < self.val_loss - self.threshold:
            self.val_loss = val_loss
            torch.save(self.network.state_dict(), self.model_file_path)

        train_acc = correct / len(train_data_loader.dataset)

        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss, correct, len(train_data_loader.dataset),
            100. * correct / len(train_data_loader.dataset)))

        return train_loss, train_acc, val_loss, val_acc

    def test(self, epoch, test_data_loader):
        self.network.eval()
        test_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = Variable(data, volatile=True).to(self.device), Variable(target).to(self.device)
            output = self.network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss

            correct += get_count_correct_preds(output, target)

            del data, target, output

        test_loss /= len(test_data_loader.dataset)
        test_acc = correct / len(test_data_loader.dataset)
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, test_loss, correct, len(test_data_loader.dataset),
            100. * correct / len(test_data_loader.dataset)))

        return  test_loss, test_acc


class JigsawModelTrainTest():

    def __init__(self, network, device, model_file_path, threshold=1e-4):
        super(JigsawModelTrainTest, self).__init__()
        self.network = network
        self.device = device
        self.model_file_path = model_file_path
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9

    def train(self, optimizer, epoch, params_max_norm, train_data_loader, val_data_loader):
        self.network.train()
        train_loss = 0
        correct = 0
        cnt_batches = 0

        for batch_idx, (data, target) in enumerate(train_data_loader):

            data, target = Variable(data).to(self.device), Variable(target).to(self.device)
            optimizer.zero_grad()
            output = self.network(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            clip_grad_norm_(self.network.parameters(), params_max_norm)

            optimizer.step()

            correct += get_count_correct_preds(output, target)
            train_loss += loss.item()
            cnt_batches += 1

            del data, target, output

        train_loss /= cnt_batches
        val_loss, val_acc = self.test(epoch, val_data_loader)

        if val_loss < self.val_loss - self.threshold:
            self.val_loss = val_loss
            torch.save(self.network.state_dict(), self.model_file_path)

        train_acc = correct / len(train_data_loader.dataset)

        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss, correct, len(train_data_loader.dataset),
            100. * correct / len(train_data_loader.dataset)))

        return train_loss, train_acc, val_loss, val_acc

    def test(self, epoch, test_data_loader):
        self.network.eval()
        test_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = Variable(data, volatile=True).to(self.device), Variable(target).to(self.device)
            output = self.network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss

            correct += get_count_correct_preds(output, target)

            del data, target, output

        test_loss /= len(test_data_loader.dataset)
        test_acc = correct / len(test_data_loader.dataset)
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, test_loss, correct, len(test_data_loader.dataset),
            100. * correct / len(test_data_loader.dataset)))

        return  test_loss, test_acc
