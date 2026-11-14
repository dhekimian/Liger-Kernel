# Minimal numerics probe: Triton fp32->bf16 rounding and libdevice math vs torch on this GPU.
import faulthandler
faulthandler.enable()
import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh as ld_tanh
from triton.language.extra.libdevice import rsqrt as ld_rsqrt


@triton.jit
def _cast(x, y, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(y + o, tl.load(x + o, mask=m).to(tl.bfloat16), mask=m)


@triton.jit
def _math(x, t, r, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    v = tl.load(x + o, mask=m)
    tl.store(t + o, ld_tanh(v), mask=m)
    tl.store(r + o, ld_rsqrt(tl.abs(v) + 1e-3), mask=m)


torch.manual_seed(0)
n = 1 << 22
x = torch.randn(n, device="cuda") * 3
grid = (triton.cdiv(n, 1024),)
y = torch.empty(n, device="cuda", dtype=torch.bfloat16)
_cast[grid](x, y, n, BLOCK=1024)
rne = x.to(torch.bfloat16)
rtz = (x.view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)
print("gpu", torch.cuda.get_device_properties(0).gcnArchName, "torch", torch.__version__, "triton", triton.__version__)
print("fp32->bf16: triton != torch(RNE):", int((y != rne).sum()), "of", n, "; triton == truncation:", int((y == rtz).sum()))
t = torch.empty_like(x)
r = torch.empty_like(x)
_math[grid](x, t, r, n, BLOCK=1024)
for name, got, ref in (("tanh", t, torch.tanh(x)), ("rsqrt", r, torch.rsqrt(x.abs() + 1e-3))):
    ulp = (got.view(torch.int32).long() - ref.view(torch.int32).long()).abs()
    print(f"{name}: elements differing from torch: {int((ulp > 0).sum())}, max ulp {int(ulp.max())}")
    print(f"{name} -> bf16 after rounding: differing {int((got.to(torch.bfloat16) != ref.to(torch.bfloat16)).sum())}")
