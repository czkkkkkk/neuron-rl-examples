"""Verify the components used by the public Neuron 2.33.1 PyTorch 2.13 recipe."""

import argparse
from importlib.metadata import version
from pathlib import Path
from subprocess import check_output

WHEELS = {
    "nki-library": "0.0.0.0a0+aca8145b",
    "torch": "2.13.0+cpu",
    "libtorch-neuronx-lite": "2.13.0.1.0.4089+71ce3449",
    "neuronx-cc": "2.28.7631.0+8f7b0cf5",
    "nki": "0.7.0+37713544100.gfede04bb",
    "torch-neuronx": "2.13.3.0.283+dc70ffa",
    "torchtitan-neuron": "0.2.2+neuron.1",
    "vllm": "0.29.0",
    "vllm-neuron": "0.29.0.1.2.0",
}
APPLICATION = {"transformers": "5.10.4", "verl": "0.9.1"}
DEBS = {
    "aws-neuronx-runtime-lib": "2.35.33.0-d2725ccb8",
    "aws-neuronx-collectives": "2.35.49.0-354ad3064",
    "aws-neuronx-tools": "2.33.38.0-f874d5618",
}


def require(name, actual, expected):
    if actual != expected:
        raise SystemExit(f"{name}: expected {expected}, found {actual}")
    print(f"{name}={actual}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="also check the loaded driver and execute a Neuron op")
    args = parser.parse_args()
    for name, expected in WHEELS.items():
        require(name, version(name), expected)
    for name, expected in APPLICATION.items():
        require(name, version(name), expected)
    for name, expected in DEBS.items():
        require(name, check_output(["dpkg-query", "-W", "-f=${Version}", name], text=True), expected)
    import hashlib
    from importlib.metadata import distribution

    kernel = distribution("nki-library").locate_file("nkilib_src/nkilib/experimental/gdn/gdn_cte_bwd.py")
    require("GDN backward SHA256", hashlib.sha256(kernel.read_bytes()).hexdigest(),
            "d481f09cf3ef043181ffefff9f1b914b63c2e63d198373b815b92affae75c377")
    if args.hardware:
        require("neuron driver", Path("/sys/module/neuron/version").read_text().strip(), "2.31.8.0")
        import torch
        import torch_neuronx  # noqa: F401

        actual = (torch.tensor([1.0, 2.0], device="neuron") + 1).cpu()
        torch.testing.assert_close(actual, torch.tensor([2.0, 3.0]))
        print("Neuron tensor addition passed")
