import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ..module.activation import act_layers

model_urls = {
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",  # noqa: E501
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",  # noqa: E501
    "shufflenetv2_1.5x": "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",  # noqa: E501
    "shufflenetv2_2.0x": "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",  # noqa: E501
}

class ChannelShuffle(nn.Module):
    #This is equivalent channel shuffle operation with custom weights so as to make the model board inferenceable.
    def __init__(self, num_channels):
        super(ChannelShuffle, self).__init__()
        assert num_channels % 4 == 0, "Number of channels must be divisible by 4"
        self.num_channels = num_channels
        
        # Define two convolutional layers to handle the output for each half
        self.conv1 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False).requires_grad_(False)
        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False).requires_grad_(False)

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            # Set weights for the first half (even indices)
            weight1 = torch.zeros(self.num_channels // 2, self.num_channels, 1, 1)
            for i in range(self.num_channels // 2):
                weight1[i, i * 2, 0, 0] = 1
            self.conv1.weight = nn.Parameter(weight1, requires_grad=False)

            # Set weights for the second half (odd indices)
            weight2 = torch.zeros(self.num_channels // 2, self.num_channels, 1, 1)
            for i in range(self.num_channels // 2):
                weight2[i, i * 2 + 1, 0, 0] = 1
      
            self.conv2.weight = nn.Parameter(weight2, requires_grad=False)

    def forward(self, x):
        x1 = self.conv1(x)  # Output for first half
        x2 = self.conv2(x)  # Output for second half
        return x1, x2

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        self.shuffle_layer = ChannelShuffle(oup)   #is channels shape of x
        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
            self.shuffle_layer = None

        else:
            self.branch_proj = None
            

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            out = torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            out =  torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        return out

    def channel_shuffle(self, x):
        #This is simple code to select even and odd channels together.
    #org implementation
        # batchsize, num_channels, height, width = x.size()
        # assert (num_channels % 4 == 0)
        # x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        # x = x.permute(1, 0, 2)  # Interleave the channels
        # x = x.reshape(2, -1, num_channels // 2, height, width)
        # return x[0], x[1]

        # assert (num_channels % 4 == 0)
        
        x0, x1 = self.shuffle_layer(x)
        return x0, x1


class ShuffleNetV2DPU(nn.Module):
    def __init__(self, stage_out_channels, load_param):
        super(ShuffleNetV2DPU, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.out_stages = (2, 3, 4)
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[0]

        #edit for 1 channels 
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+1]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))
        
        if load_param == True:
            print("load param...")
            self._initialize_weights()
        else:
            print('The backbone wts are randomly intialized and no pretrained is used.')
            

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self):
        print("initialize_weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        missing_keys, unexpected_keys = self.load_state_dict(torch.load("./model/backbone/backbone.pth", map_location=device), strict = False)

        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)

