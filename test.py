'''
This file implements a basic test for switched convolution blocks using the CIFAR-100 dataset.
It builds up a basic 4-layer deep test module using switched blocks and trains it.

Various statistics are outputted to prove that the switching mechanism is actually working (and being trained)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init
from .switched_conv import SwitchedConv2d


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class DynamicConvTestModule(nn.Module):
    def __init__(self):
        super(DynamicConvTestModule, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=True)
        self.conv1 = SwitchedConv2d(
            16, 32, 3, stride=2, pads=1, num_convs=4, initial_temperature=10
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = SwitchedConv2d(
            32,
            64,
            3,
            stride=2,
            pads=1,
            att_kernel_size=3,
            att_padding=1,
            num_convs=8,
            initial_temperature=10,
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = SwitchedConv2d(
            64,
            128,
            3,
            stride=2,
            pads=1,
            att_kernel_size=3,
            att_padding=1,
            num_convs=16,
            initial_temperature=10,
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(128 * 4 * 4, 256)
        self.dense2 = nn.Linear(256, 100)
        self.softmax = nn.Softmax(-1)

    def set_temp(self, temp):
        self.conv1.set_attention_temperature(temp)
        self.conv2.set_attention_temperature(temp)
        self.conv3.set_attention_temperature(temp)

    def forward(self, x):
        x = self.init_conv(x)
        x, att1 = self.conv1(x, output_attention_weights=True)
        x = self.relu(self.bn1(x))
        x, att2 = self.conv2(x, output_attention_weights=True)
        x = self.relu(self.bn2(x))
        x, att3 = self.conv3(x, output_attention_weights=True)
        x = self.relu(self.bn3(x))
        atts = [att1, att2, att3]
        usage_hists = []
        mean = 0
        for a in atts:
            m, u = compute_attention_specificity(a)
            mean += m
            usage_hists.append(u)
        mean /= 3

        x = x.flatten(1)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        # Compute metrics across attention weights.

        return self.softmax(x), mean, usage_hists


def test_dynamic_conv():
    writer = SummaryWriter()
    dataset = datasets.ImageFolder(
        "E:\\data\\cifar-100-python\\images\\train",
        transforms.Compose(
            [
                transforms.Resize(32, 32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )
    batch_size = 256
    temperature = 30
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    device = torch.device("cuda:0")
    net = DynamicConvTestModule()
    net = net.to(device)
    net.set_temp(temperature)
    initialize_weights(net)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Load state, where necessary.
    """
    netstate, optimstate = torch.load("test_net.pth")
    net.load_state_dict(netstate)
    optimizer.load_state_dict(optimstate)
    """

    criterion = nn.CrossEntropyLoss()
    step = 0
    running_corrects = 0
    running_att_mean = 0
    running_att_hist = None
    for e in range(300):
        tq = tqdm.tqdm(loader)
        for batch, labels in tq:
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, att_mean, att_usage_hist = net.forward(batch)
            running_att_mean += att_mean
            if running_att_hist is None:
                running_att_hist = att_usage_hist
            else:
                for i in range(len(att_usage_hist)):
                    running_att_hist[i] = torch.cat(
                        [running_att_hist[i], att_usage_hist[i]]
                    ).flatten()
            loss = criterion(logits, labels)
            loss.backward()

            '''
            if step % 50 == 0:
                c1_grad_avg = sum(
                    [
                        m.weight.grad.abs().mean().item()
                        for m in net.conv1.conv_list._modules.values()
                    ]
                ) / len(net.conv1.conv_list._modules)
                c1a_grad_avg = (
                    net.conv1.attention_conv1.weight.grad.abs().mean()
                    + net.conv1.attention_conv2.weight.grad.abs().mean()
                ) / 2
                c2_grad_avg = sum(
                    [
                        m.weight.grad.abs().mean().item()
                        for m in net.conv2.conv_list._modules.values()
                    ]
                ) / len(net.conv2.conv_list._modules)
                c2a_grad_avg = (
                    net.conv2.attention_conv1.weight.grad.abs().mean()
                    + net.conv2.attention_conv2.weight.grad.abs().mean()
                ) / 2
                c3_grad_avg = sum(
                    [
                        m.weight.grad.abs().mean().item()
                        for m in net.conv3.conv_list._modules.values()
                    ]
                ) / len(net.conv3.conv_list._modules)
                c3a_grad_avg = (
                    net.conv3.attention_conv1.weight.grad.abs().mean()
                    + net.conv3.attention_conv2.weight.grad.abs().mean()
                ) / 2
                writer.add_scalar("c1_grad_avg", c1_grad_avg, global_step=step)
                writer.add_scalar("c2_grad_avg", c2_grad_avg, global_step=step)
                writer.add_scalar("c3_grad_avg", c3_grad_avg, global_step=step)
                writer.add_scalar("c1a_grad_avg", c1a_grad_avg, global_step=step)
                writer.add_scalar("c2a_grad_avg", c2a_grad_avg, global_step=step)
                writer.add_scalar("c3a_grad_avg", c3a_grad_avg, global_step=step)
            '''

            optimizer.step()
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == labels.data)
            if step % 50 == 0:
                print(
                    "Step: %i, Loss: %f, acc: %f, att_mean: %f"
                    % (
                        step,
                        loss.item(),
                        running_corrects / (50.0 * batch_size),
                        running_att_mean / 50.0,
                    )
                )
                writer.add_scalar("Loss", loss.item(), global_step=step)
                writer.add_scalar(
                    "Accuracy", running_corrects / (50.0 * batch_size), global_step=step
                )
                writer.add_scalar("Att Mean", running_att_mean / 50, global_step=step)
                for i in range(len(running_att_hist)):
                    writer.add_histogram(
                        "Att Hist %i" % (i,), running_att_hist[i], global_step=step
                    )
                writer.flush()
                running_corrects = 0
                running_att_mean = 0
                running_att_hist = None
            if step % 1000 == 0:
                temperature = max(temperature - 1, 1)
                net.set_temp(temperature)
                print("Temperature drop. Now: %i" % (temperature,))
            step += 1
        torch.save((net.state_dict(), optimizer.state_dict(), amp.state_dict()), "test_net.pth")


if __name__ == "__main__":
    test_dynamic_conv()