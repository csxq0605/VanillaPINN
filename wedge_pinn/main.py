"""Entry point. Every Config field is a command-line flag, e.g.

    python main.py --adam_iters 20000 --lbfgs_iters 3000 --tag baseline
"""
import argparse
from dataclasses import fields

from train import Config, Trainer


def parse():
    ap = argparse.ArgumentParser()
    for f in fields(Config):
        if f.type in (bool, "bool"):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type, f.type) if isinstance(f.type, str) else f.type
            ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    return Config(**vars(ap.parse_args()))


if __name__ == "__main__":
    Trainer(parse()).train()
