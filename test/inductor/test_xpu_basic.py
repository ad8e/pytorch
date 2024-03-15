# Owner(s): ["module: inductor"]
import torch

try:
    from .test_torchinductor import check_model_gpu, TestCase
except ImportError:
    from test_torchinductor import check_model_gpu, TestCase

# TODO:
# This is a temp test case for first Intel GPU Inductor PR. It test basic functionality of Intel GPU backend.
# We are reusing test case in test/inductor/*  step by step.
# Will be removed this file after Intel GPU backend can pass tests in test/inductor/*
# 
class XpuBasicTests(TestCase):
    common = check_model_gpu
    device = "xpu"

    def test_add(self):
        def fn(a, b):
            return a + b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_sub(self):
        def fn(a, b):
            return a - b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_mul(self):
        def fn(a, b):
            return a * b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))

    def test_div(self):
        def fn(a, b):
            return a / b

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_XPU

    if HAS_XPU:
        run_tests(needs="filelock")
