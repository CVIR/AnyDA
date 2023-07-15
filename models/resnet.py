import torch.nn as nn
import math
from utils.config import FLAGS
from builtins import id
from boto.gs.acl import ID
depth_mult = FLAGS.depth_mult_range[-1]
import math
from .slimmable_ops import USBatchNorm2d, TwoInputSequential
from .slimmable_ops import USConv2d, USLinear, make_divisible
from utils.config import FLAGS
import torch
from .dsbn import DomainSpecificBatchNorm2d

print("Total gpus:",torch.cuda.device_count())
gpus = torch.cuda.device_count()
#id = [0]*gpus
#print(id)
#depth = [1, 1, 1, 1]

class Block(nn.Module):
    def __init__(self, inp, outp, stride, tmp_ratio=1.0):
        super(Block, self).__init__()
        assert stride in [1, 2]  
        self.skip = False   
        midp = make_divisible(outp // 4)
        expand_ratio = 0.25
        layers = [
            USConv2d(inp, midp, 1, 1, 0, bias=False, ratio=[tmp_ratio, expand_ratio]),
            DomainSpecificBatchNorm2d(num_features=midp, num_classes=2, ratio=expand_ratio),
            nn.ReLU(inplace=True),

            USConv2d(midp, midp, 3, stride, 1, bias=False, ratio=[expand_ratio, expand_ratio]),
            DomainSpecificBatchNorm2d(num_features=midp, num_classes=2, ratio=expand_ratio),
            nn.ReLU(inplace=True),

            USConv2d(midp, outp, 1, 1, 0, bias=False, ratio=[expand_ratio, 1]),
            DomainSpecificBatchNorm2d(num_features=outp, num_classes=2),
        ]
        self.body = TwoInputSequential(*layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = TwoInputSequential(
                USConv2d(inp, outp, 1, stride=stride, bias=False, ratio=[tmp_ratio, 1]),
                DomainSpecificBatchNorm2d(num_features=outp, num_classes=2),
            )
        self.post_relu = nn.ReLU(inplace=True)
        
    def forward(self, x, dom=[0]):
        self.dom=dom
        global id
#         if id>15:
#             id-=16
        dev = torch.cuda.current_device()
        #print("depth in block:",depth)
        #print("Blk dev: ",dev)
        #print("Block {} Input: {}".format(id,x.shape))
        if id[dev] in range(0, (0+depth[0])) or id[dev] in range(3, (3+depth[1])) or id[dev] in range(7, (7+depth[2])) or id[dev] in range(13, (13+depth[3])):
            #print("Block {} In: {}".format(id,x.shape))
            self.skip = False
            if self.residual_connection:
                res = self.body(x)
                res += x
            else:
                res = self.body(x)
                res += self.shortcut(x)
            res = self.post_relu(res)
            #print("Block {} Out: {}".format(id,res.shape))
            id[dev] += 1
            return res
        else:
            self.skip = True
            #print("Block {} is skipped".format(id))
            id[dev] += 1
            return x


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()
        self.features_ini = []
        self.features_blks = []
        self.features_last = []
        # head
        assert input_size % 32 == 0
        self.depth_mult = None

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        init_channel = 64
        channels = make_divisible(init_channel)
        self.block_setting = self.block_setting_dict[FLAGS.depth]
        feats = [64, 128, 256, 512]
        self.features_ini = TwoInputSequential(
                                            USConv2d(3, channels, 7, 2, 3, bias=False, us=[False, True], ratio=[1, 0.25]),
                                            DomainSpecificBatchNorm2d(num_features=channels, num_classes=2, ratio=0.25),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(3, 2, 1),
                                        )

        # body
        for stage_id, n in enumerate(self.block_setting):
            outp = make_divisible(feats[stage_id] * 4)
            for i in range(n):
                if i == 0 and stage_id != 0:
                    self.features_blks.append(Block(channels, outp, 2))
                elif i == 0 and stage_id == 0:
                    self.features_blks.append(Block(channels, outp, 1, tmp_ratio=0.25))
                else:
                    self.features_blks.append(Block(channels, outp, 1))
                #self.id += self.id
                channels = outp

        # avg_pool_size = input_size // 32
        # self.features.append(nn.AvgPool2d(avg_pool_size))
        self.features_last.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # make it nn.Sequential
        # self.features_ini = nn.Sequential(*self.features_ini)
        self.features_blks = TwoInputSequential(*self.features_blks)
        self.features_last = TwoInputSequential(*self.features_last)

        # classifier
        self.outp = channels
        self.classifier = TwoInputSequential(
            USLinear(self.outp, num_classes, us=[True, False])
        )

        if FLAGS.reset_parameters:
            self.reset_parameters()
    def forward(self, x, dom=[0]):
        global depth
        global id
        self.dom=dom
        id = [0]*gpus
        #print("depth_mult",self.depth_mult)
        self.block_setting = [int(val*self.depth_mult) for val in self.block_setting_dict[FLAGS.depth]]
        depth = self.block_setting
        #print("depth",depth)
        x = self.features_ini(x, dom=self.dom)
        #print("Model global id: ",id)
        x = self.features_blks(x, dom=self.dom)
        x = self.features_last(x)
        last_dim = x.size()[1]
        #print("last dim b4 classify:", last_dim)
        x = x.view(-1, last_dim)
        #self.outp=last_dim
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()