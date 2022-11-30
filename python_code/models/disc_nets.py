import torch
import torch.nn as nn


class DiscIEGMNet_cbr_fconv(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=6,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(3, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=5,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(2, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(2, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(2, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=4,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(8, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=37,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(1, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1),
        )

    def forward(self, x):
        # x = x.squeeze(dim=-1)
        conv1_output = self.conv1(x)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        # conv5_output = conv5_output.flatten(1)

        fc1_output = self.fc1(conv5_output)
        fc2_output = self.fc2(fc1_output)
        # return fc2_output.squeeze(-1)
        # return fc2_output[:, :, 0]
        fc2_output = torch.flatten(fc2_output, 1)
        return fc2_output


class DiscIEGMNet_cbr(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=6,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(3, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=5,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(2, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(2, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=4,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(2, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=4,
                      stride=2, padding=0, bias=False),
            nn.BatchNorm1d(8, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=296, out_features=1, bias=False),
            # nn.Conv1d(in_channels=8, out_channels=1, kernel_size=37,
            #           stride=1, padding=0, bias=False),
            nn.BatchNorm1d(1, affine=True, track_running_stats=True,
                           eps=1e-5, momentum=0.1),
            nn.ReLU(True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1, out_features=2),
        )

    def forward(self, x):
        conv1_output = self.conv1(x)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.flatten(1)

        fc1_output = self.fc1(conv5_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output
