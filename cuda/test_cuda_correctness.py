import functools
import unittest
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import switched_conv_cuda_naive
from torch.autograd import gradcheck
from tqdm import tqdm

from weighted_conv_extension.switched_conv_naive.switched_conv_hard_routing import SwitchedConvHardRoutingFunction, \
    SwitchedConvHardRouting
from weighted_conv_extension.switched_conv_naive.torch_switched_conv import SwitchedConv

EPS = 1e-3
def abs_mean(t1, t2):
    return torch.max(torch.abs(t1 - t2))


def cnv_back(storage, mod, gi, go):
    import pydevd
    pydevd.settrace(suspend=False, trace_only_current_thread=True)
    storage['grad_out'] = gi[0]


def index_2d(input, index):
    index = index.repeat(1,input.shape[1],1,1)
    e = torch.eye(input.shape[-1], device=input.device)
    result = e[index] * input
    return result.sum(-1)


class TestCpuCorrectness(unittest.TestCase):
    def test_conv1x1_no_bias(self):
        inp = torch.randn(1, 32, 32, 32, dtype=torch.double).to('cuda')*1000+5
        weight = torch.randn(32, 32, 8, 1, 1, dtype=torch.double).to('cuda')*1000+5
        mask = torch.zeros((8,32,32), dtype=torch.int).to('cuda')
        bias = torch.zeros(32, dtype=torch.double).to('cuda')
        out1 = F.conv2d(inp.cpu(), weight[:,:,0,:,:].cpu(), bias, stride=1, padding=0)
        out2 = switched_conv_cuda_naive.forward(inp, mask, weight, bias, 1).cpu()
        self.assertLess(abs_mean(out1, out2), EPS)

    def test_conv1x1(self):
        inp = torch.randn(8, 32, 32, 32, dtype=torch.double).to('cuda')*1000+5
        weight = torch.randn(64, 32, 8, 1, 1, dtype=torch.double).to('cuda')*1000+5
        mask = torch.zeros((8,32,32), dtype=torch.int).to('cuda')
        bias = torch.randn(64, dtype=torch.double).to('cuda')*1000+5
        out1 = F.conv2d(inp.cpu(), weight[:,:,0,:,:].cpu(), bias=bias.cpu(), stride=1, padding=0)
        out2 = switched_conv_cuda_naive.forward(inp, mask, weight, bias, 1).cpu()
        self.assertLess(abs_mean(out1, out2), EPS)

    def test_conv3x3_breadth_1(self):
        inp = torch.randn(8, 32, 32, 32, dtype=torch.double).to('cuda')*1000+5
        weight = torch.randn(64, 32, 1, 3, 3, dtype=torch.double).to('cuda')*1000+5
        mask = torch.zeros((8,32,32), dtype=torch.int).to('cuda')
        bias = torch.randn(64).to('cuda')*1000+5
        out1 = F.conv2d(inp.cpu(), weight[:,:,0,:,:].cpu(), bias=bias.cpu(), stride=1, padding=1)
        out2 = switched_conv_cuda_naive.forward(nn.ZeroPad2d(1)(inp), mask, weight, bias, 1).cpu()
        self.assertLess(abs_mean(out1, out2), EPS)

    def test_convnxn(self):
        for n, p in zip([3,5,7], [1,2,3]):
            inp = torch.randn(8, 32, 32, 32, dtype=torch.double).to('cuda')
            weight = torch.randn(64, 32, 8, n, n, dtype=torch.double).to('cuda')
            mask = torch.zeros((8,32,32), dtype=torch.int).to('cuda')
            bias = torch.randn(64, dtype=torch.double).to('cuda')
            out1 = F.conv2d(inp.cpu(), weight[:,:,0,:,:].cpu(), bias=bias.cpu(), stride=1, padding=p)
            out2 = switched_conv_cuda_naive.forward(nn.ZeroPad2d(p)(inp), mask, weight, bias, 1).cpu()
            self.assertLess(abs_mean(out1, out2), EPS)

    def test_awkward_channels(self):
        for i, o in zip([31,57,73], [12,96,7]):
            inp = torch.randn(8, i, 32, 32, dtype=torch.double).to('cuda')
            weight = torch.randn(o, i, 8, 3, 3, dtype=torch.double).to('cuda')
            mask = torch.zeros((8,32,32), dtype=torch.int).to('cuda')
            bias = torch.randn(o, dtype=torch.double).to('cuda')
            out1 = F.conv2d(inp.cpu(), weight[:,:,0,:,:].cpu(), bias=bias.cpu(), stride=1, padding=1)
            out2 = switched_conv_cuda_naive.forward(nn.ZeroPad2d(1)(inp), mask, weight, bias, 1).cpu()
            self.assertLess(abs_mean(out1, out2), EPS)

    def test_conv3x3_strided(self):
        for stride in [2,4]:
            inp = torch.randn(8, 32, 32, 32, dtype=torch.double).to('cuda')
            weight = torch.randn(64, 32, 8, 5, 5, dtype=torch.double).to('cuda')
            mask = torch.zeros((8,32//stride,32//stride), dtype=torch.int).to('cuda')
            bias = torch.randn(64, dtype=torch.double).to('cuda')
            out1 = F.conv2d(inp.cpu(), weight[:,:,0,:,:].cpu(), bias.cpu(), padding=2, stride=stride)
            out2 = switched_conv_cuda_naive.forward(nn.ZeroPad2d(2)(inp), mask, weight, bias, stride).cpu()
            self.assertLess(abs_mean(out1, out2), EPS)

    def test_convnxn_random_same_weights(self):
        for n, p in zip([3,5,7], [1,2,3]):
            inp = torch.randn(8, 32, 32, 32, dtype=torch.double).to('cuda')*7
            weight = torch.randn(64, 32, 1, n, n, dtype=torch.double).repeat(1,1,8,1,1).to('cuda')*1000
            mask = torch.randint(low=0, high=7, size=(8,32,32), dtype=torch.int).to('cuda')
            bias = torch.randn(64, dtype=torch.double).to('cuda')
            out1 = F.conv2d(inp.cpu(), weight[:,:,0,:,:].cpu(), bias=bias.cpu(), stride=1, padding=p)
            out2 = switched_conv_cuda_naive.forward(nn.ZeroPad2d(p)(inp), mask, weight, bias, 1).cpu()
            self.assertLess(abs_mean(out1, out2), EPS)

    def test_convnxn_with_torch_switched_conv(self):
        swcnv = SwitchedConv(128,64,3,8,padding=1,hard_attention=True,include_coupler=False)
        cuda_swcnv = SwitchedConvHardRouting(128,64,3,breadth=8)
        sd = swcnv.state_dict()
        csd = {
            'weight': torch.stack([sd[f'weights.{i}'] for i in range(8)], dim=2),
            'bias': sd['bias']
        }
        cuda_swcnv.load_state_dict(csd)
        cuda_swcnv = cuda_swcnv.to('cuda')
        for n, p in zip([3,5,7], [1,2,3]):
            inp = torch.randn(8, 128, 32, 32).to('cuda')
            sel = torch.randn(8,8,32,32).to('cuda')
            out1 = swcnv(inp.cpu(), sel.cpu())
            out2 = cuda_swcnv(inp, sel)
            self.assertLess(abs_mean(out1, out2.cpu()), EPS)

    def test_conv3x3_awkward_shapes(self):
        for w, h in zip([57, 33, 5, 9, 211], [3, 57, 100, 101, 211]):
            inp = torch.randn(8, 32, h, w, dtype=torch.double).to('cuda')*1000+5
            weight = torch.randn(64, 32, 8, 3, 3, dtype=torch.double).to('cuda')*1000+5
            mask = torch.zeros((8,h,w), dtype=torch.int).to('cuda')
            bias = torch.randn(64, dtype=torch.double).to('cuda')*1000+5
            out1 = F.conv2d(inp.cpu(), weight[:,:,0,:,:].cpu(), bias=bias.cpu(), stride=1, padding=1)
            out2 = switched_conv_cuda_naive.forward(nn.ZeroPad2d(1)(inp), mask, weight, bias, 1).cpu()
            self.assertLess(abs_mean(out1, out2), EPS)


    def test_perf_forward(self):
        b=32
        swcnv = SwitchedConv(64,64,5,b,padding=2,hard_attention=True,include_coupler=False,stride=2).cuda()
        cuda_swcnv = SwitchedConvHardRouting(64,64,5,breadth=b,stride=2)
        sd = swcnv.state_dict()
        csd = {
            'weight': torch.stack([sd[f'weights.{i}'] for i in range(b)], dim=2),
            'bias': sd['bias']
        }
        cuda_swcnv.load_state_dict(csd, strict=False)
        cuda_swcnv = cuda_swcnv.to('cuda')

        print("Testing standard cudnn implementation..")
        # Perf: b_8=3.96it/s  b_16=3.24it/s  b_32=2.03it/s
        for k in tqdm(range(30)):
            inp = torch.randn(32, 64, 128, 128).to('cuda')
            sel = torch.randn(32,b,64,64).to('cuda')
            out1 = swcnv(inp, sel)
            torch.cuda.synchronize()

        print("Testing custom implementation")
        # Perf: b_8=3.11it/s  b_16=2.98it/s  b_32=2.52it/s
        for k in tqdm(range(30)):
            inp = torch.randn(32, 64, 128, 128).to('cuda')
            sel = torch.randn(32,b,128,128).to('cuda')
            out2 = cuda_swcnv(inp, sel)
            torch.cuda.synchronize()

    def test_performance(self):
        inp = torch.randn(16, 64, 256, 256).to('cuda')
        weight = torch.randn(64, 64, 3, 3, 8).to('cuda')
        bias = torch.randn(64).to('cuda')
        mask = torch.randint(low=0, high=8, size=(256, 256), dtype=torch.int).to('cuda')
        RANGE = 30

        for j in range(RANGE):
            if j == 2:
                conv_time = time()
            o = [F.conv2d(inp, weight[:,:,:,:,0], bias, 1, 1) for n in range(8)]
            #index_2d(torch.stack(o, dim=-1), mask.unsqueeze(0).unsqueeze(0).repeat(16, 1, 1, 1).long())
            torch.cuda.synchronize()
        conv_time = time() - conv_time
        print("CUDNN conv:", conv_time)

    def test_conv_backwards(self):
        w = 128
        h = 128
        for cin, cout in zip([32, 16, 32, 27], [32, 32, 16, 33]):
            for ks in [1,3,5]:
                for s in [1,2,4]:
                    rand_scale = torch.randn((8,cout,h//s,w//s)).to('cuda').double()
                    inp = torch.randn((8,cin,h,w), requires_grad=True).to('cuda').double()*1000+5
                    conv = nn.Conv2d(cin,cout,ks, bias=True, padding=ks//2, stride=s).to('cpu').double()
                    conv_store = {}
                    conv.register_backward_hook(functools.partial(cnv_back, conv_store))
                    (conv(inp.cpu())*rand_scale.cpu()).mean().backward()

                    inp_other = inp.detach().clone()
                    wconv_store = {}
                    wconv = SwitchedConvHardRouting(cin, cout, ks, bias=True, stride=s, breadth=8).double().to('cuda')
                    wconv.load_weights_from_conv(conv)
                    wconv.register_backward_hook(functools.partial(cnv_back, wconv_store))
                    (wconv(inp_other, torch.zeros((8, 8, h//s, w//s)).to('cuda')) * rand_scale).mean().backward()
                    self.assertLess(abs_mean(conv_store['grad_out'], wconv_store['grad_out'].cpu()), EPS)
                    self.assertLess(abs_mean(conv.weight.grad, wconv.weight.grad[:,:,0,:,:].cpu()), EPS)
                    self.assertLess(abs_mean(conv.bias.grad, wconv.bias.grad.cpu()), EPS)
                    wconv.zero_grad()
                    conv.zero_grad()

    def test_conv_backwards_against_sim(self):
        w = 128
        h = 128
        for cin, cout in zip([32, 16, 32, 27], [32, 32, 16, 33]):
            for s in [1,2,4]:
                for ks in [1,3,5]:
                    rand_scale = torch.randn((8,cout,h//s,w//s)).to('cuda').double()
                    in_src = torch.randn((8,cin,h,w)).to('cuda').double()*1000+5
                    sel_src = torch.rand((8,8,h//s,w//s), dtype=torch.double, requires_grad=False).to('cuda')
                    inp = nn.Parameter(in_src)
                    sel = nn.Parameter(sel_src)
                    conv = SwitchedConvHardRouting(cin, cout, ks, bias=True, stride=s, breadth=8, sim_mode=True).double().to('cuda')
                    out1 = conv(inp, sel)
                    (out1*rand_scale).mean().backward()

                    inp_other = nn.Parameter(in_src.detach().clone())
                    sel_other = nn.Parameter(sel_src.detach().clone())
                    wconv = SwitchedConvHardRouting(cin, cout, ks, bias=True, stride=s, breadth=8).double().to('cuda')
                    wconv.load_state_dict(conv.state_dict())
                    out2 = wconv(inp_other,sel_other)
                    (out2*rand_scale).mean().backward()
                    self.assertLess(abs_mean(out1, out2), EPS)
                    self.assertLess(abs_mean(inp.grad, inp_other.grad), EPS)
                    self.assertLess(abs_mean(sel.grad, sel_other.grad), EPS)
                    self.assertLess(abs_mean(conv.weight.grad, wconv.weight.grad), EPS)
                    self.assertLess(abs_mean(conv.bias.grad, wconv.bias.grad), EPS)
                    wconv.zero_grad()
                    conv.zero_grad()

    def test_gradcheck(self):
        inp = (torch.randn((1,8,16,16), device='cuda', dtype=torch.double),
               torch.zeros((1,8,16,16), device='cuda', dtype=torch.double),
               torch.randn((8,8,8,3,3), device='cuda', dtype=torch.double),
               torch.randn((8,), device='cuda', dtype=torch.double, requires_grad=True))
        test = gradcheck(SwitchedConvHardRoutingFunction.apply, inp, atol=1e-3, raise_exception=True)
        print(test)


    def test_backwards_performance(self):
        s = 1
        h, w = 256, 256
        chan = 32
        batch = 16
        ks = 3

        inp = torch.randn(batch, chan, h, w).to('cuda') * 1000 + 5
        conv = nn.Conv2d(chan, chan, ks, bias=True, padding=ks//2-1, stride=s).to('cuda')
        conv_store = {}
        conv.register_backward_hook(functools.partial(cnv_back, conv_store))
        start = time()
        for i in range(10):
            conv(inp).mean().backward()
        torch.cuda.synchronize()
        print("CUDNN: ", (time()-start))

        wconv_store = {}
        wconv = SwitchedConvHardRouting(chan, chan, ks, bias=True, stride=s, breadth=8).to('cuda')
        wconv.register_backward_hook(functools.partial(cnv_back, wconv_store))
        start = time()
        for i in tqdm(range(10)):
            wconv(inp, torch.zeros((batch, 8, h, w)).to('cuda')).mean().backward()
        torch.cuda.synchronize()
        print("Custom: ", (time()-start))

    def test_component_wise_performance(self):
        inp = torch.randn((16,32,260,260),device='cuda')
        grd = torch.randn((16,32,256,256),device='cuda')
        kern = torch.randn((32,32,8,5,5),device='cuda')
        bias = torch.randn((32,),device='cuda')
        mask = torch.zeros((16,256,256),dtype=torch.int,device='cuda')

        # Last checkpoint:
        # Forward: 2.9
        # Gradin: 1.8
        # GradK: 2.9

        start = time()
        for j in tqdm(range(50)):
            switched_conv_cuda_naive.forward(inp, mask, kern, bias, 1)
        torch.cuda.synchronize()
        print("Forward:", (time()-start))
        '''
        start = time()
        for j in tqdm(range(50)):
            switched_conv_cuda_naive.bench_grad_input(inp, grd, mask, kern, bias, 1)
        torch.cuda.synchronize()
        print("Grad Input:", (time()-start))

        start = time()
        for j in tqdm(range(50)):
            switched_conv_cuda_naive.bench_grad_kernel(inp, grd, mask, kern, bias, 1)
        torch.cuda.synchronize()
        print("Grad Kernel:", (time()-start))
        '''



if __name__ == '__main__':
    unittest.main()
