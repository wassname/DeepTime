"""
Modifie from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTimePlus.py
"""
from tsai.models.InceptionTimePlus import Conv, Module, noop, Integral, nn, is_listy, SimpleSelfAttention, Concat, SqueezeExciteBlock, Norm, BN1d, delegates, ConvBlock, Add, np, random, ifnone, OrderedDict, Flatten, SigmoidRange, LinBnDrop, GACP1d, GAP1d, named_partial, F, torch, CausalConv1d, Noop

Conv = named_partial('Conv', ConvBlock, norm=None, act=None, padding='causal')
# CausalConvBlock = named_partial('CausalConv', ConvBlock, padding='causal')

class CausalMaxPool1d(torch.nn.MaxPool1d):
    def __init__(self, ks, stride=1, padding=0, dilation=1):
        super().__init__(kernel_size=ks, stride=stride, padding=0, dilation=dilation)
        self.__padding = (ks - 1) * dilation
    def forward(self, input):
        return super().forward(F.pad(input, (self.__padding, 0)))

class InceptionModulePlus(Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True, padding='causal', coord=False, separable=False, dilation=1, stride=1, conv_dropout=0., sa=False, se=None,
                 norm='Batch', zero_norm=False, bn_1st=True, act=nn.ReLU, act_kwargs={}):

        dilation = max(1, dilation)

        if not (is_listy(ks) and len(ks) == 3):
            if isinstance(ks, Integral): ks = [ks // (2**i) for i in range(3)]
            ks = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in ks]  # ensure odd ks for padding='same'

        bottleneck = False if ni == nf else bottleneck
        self.bottleneck = Conv(ni, nf, 1, coord=coord, bias=False) if bottleneck else noop #
        self.convs = nn.ModuleList()
        for i in range(len(ks)): self.convs.append(Conv(nf if bottleneck else ni, nf, ks[i], padding=padding, coord=coord, separable=separable,
                                                         dilation=dilation**i, stride=stride, bias=False))
        self.mp_conv = nn.Sequential(*[Conv(ni, nf, 1, coord=coord, bias=False)])
        self.concat = Concat()
        if norm is not None:
            self.norm = Norm(nf * 4, norm=norm, zero_norm=zero_norm)
        else:
            self.norm = noop
        self.conv_dropout = nn.Dropout(conv_dropout) if conv_dropout else noop
        self.sa = SimpleSelfAttention(nf * 4) if sa else noop
        self.act = act(**act_kwargs) if act else noop
        self.se = nn.Sequential(SqueezeExciteBlock(nf * 4, reduction=se), BN1d(nf * 4)) if se else noop

        self._init_cnn(self)
    
    def _init_cnn(self, m):
        if getattr(self, 'bias', None) is not None: nn.init.constant_(self.bias, 0)
        if isinstance(self, (nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear)): nn.init.kaiming_normal_(self.weight)
        for l in m.children(): self._init_cnn(l)

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        x = self.concat([l(x) for l in self.convs] + [self.mp_conv(input_tensor)])
        x = self.norm(x)
        x = self.conv_dropout(x)
        x = self.sa(x)
        x = self.act(x)
        x = self.se(x)
        return x


@delegates(InceptionModulePlus.__init__)
class InceptionBlockPlus(Module):
    def __init__(self, ni, nf, residual=True, depth=6, coord=False, norm=None, zero_norm=False, act=nn.ReLU, act_kwargs={}, sa=False, se=None, dilation=1,
                 stoch_depth=1., **kwargs):
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut, self.act = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModulePlus(ni if d == 0 else nf * 4, nf, coord=coord, norm=norm,
                                                      zero_norm=zero_norm if d % 3 == 2 else False,
                                                      act=act if d % 3 != 2 else None, act_kwargs=act_kwargs,
                                                      sa=sa if d % 3 == 2 else False,
                                                      se=se if d % 3 != 2 else None,
                                                      dilation=dilation*d*(dilation>1),
                                                      **kwargs))
            if self.residual and d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                if norm is not None:
                    n = Norm(n_in, norm=norm)
                else:
                    n = Noop
                self.shortcut.append(n if n_in == n_out else ConvBlock(n_in, n_out, 1, coord=coord, bias=False, norm=norm, padding='causal', act=None))
                self.act.append(act(**act_kwargs))
        self.add = Add()
        if stoch_depth != 0: keep_prob = np.linspace(1, stoch_depth, depth)
        else: keep_prob = np.array([1] * depth)
        self.keep_prob = keep_prob

    def forward(self, x):
        res = x
        for i in range(self.depth):
            if self.keep_prob[i] > random.random() or not self.training:
                x = self.inception[i](x)
            if self.residual and i % 3 == 2:
                res = x = self.act[i//3](self.add(x, self.shortcut[i//3](res)))
        return x

# Cell
@delegates(InceptionModulePlus.__init__)
class CausalInceptionTimePlus(nn.Sequential):
    def __init__(self, c_in, c_out, seq_len=None, nf=32, nb_filters=None,
                 flatten=False, concat_pool=False, fc_dropout=0., bn=False, y_range=None, custom_head=None, **kwargs):

        if nb_filters is not None: nf = nb_filters
        else: nf = ifnone(nf, nb_filters) # for compatibility
        backbone = InceptionBlockPlus(c_in, nf, **kwargs)

        #head
        self.head_nf = nf * 4
        self.c_out = c_out
        self.seq_len = seq_len
        if custom_head: head = custom_head(self.head_nf, c_out, seq_len)
        else: head = self.create_head(self.head_nf, c_out, seq_len, flatten=flatten, concat_pool=concat_pool,
                                      fc_dropout=fc_dropout, bn=bn, y_range=y_range)

        layers = OrderedDict([('backbone', nn.Sequential(backbone)), ('head', nn.Sequential(head))])
        super().__init__(layers)

        self.calc_receptive_field(kwargs.get('ks'), kwargs.get('depth'), kwargs.get('dilation', 1))

    def calc_receptive_field(self, ks, depth, dilation):       
        # receptive fields vs R
        ks=np.array(ks)
        d=np.array([dilation**i for i in range(3)])
        rf = (ks-1)*d*depth

        dilations = np.array([max(1, d*dilation) for d in range(depth)])
        d=np.array([dilations**i for i in range(3)]).T
        rf = ((ks-1)*d).sum(0)
        print(f"receptive field {rf}={ks-1}*{d}")

    def create_head(self, nf, c_out, seq_len, flatten=False, concat_pool=False, fc_dropout=0., bn=False, y_range=None):
        if flatten:
            nf *= seq_len
            layers = [Flatten()]
        else:
            if concat_pool: nf *= 2
            layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)
