"""Check the three-step Qwen3.8-27B GSM8K recipe from its train.log."""

import argparse
import math
import re
from pathlib import Path

DELIVERY = re.compile(r"delivered bucket (\d+)/(\d+) to all (\d+) receivers")
METRIC = re.compile(r"(actor/grad_norm|critic/rewards/mean):([^\s]+)")


def verify(log):
    steps = {}
    transfers = 0
    with log.open(errors="replace") as stream:
        for line in stream:
            match = DELIVERY.search(line)
            if match:
                bucket, count, recipients = map(int, match.groups())
                if (bucket, count, recipients) != (transfers % 401 + 1, 401, 32):
                    raise ValueError(f"Incomplete or unexpected policy delivery: {match[0]}")
                transfers += 1
            match = re.search(r"\bstep:(\d+)\b", line)
            if match:
                metrics = dict(METRIC.findall(line))
                if len(metrics) == 2:
                    step = int(match[1])
                    grad, reward = float(metrics["actor/grad_norm"]), float(metrics["critic/rewards/mean"])
                    if step in steps or not math.isfinite(grad) or grad <= 0 or not 0 <= reward <= 1:
                        raise ValueError(f"Unexpected training metrics at step {step}: {metrics}")
                    steps[step] = (grad, reward)
    if list(steps) != [1, 2, 3] or transfers != 4 * 401:
        raise ValueError(f"Expected steps 1, 2, 3 and four full policy deliveries; got {list(steps)} and {transfers} buckets")
    print("PASS: three optimizer updates; four complete 401-bucket policy deliveries to 32 workers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    try:
        verify(parser.parse_args().log)
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")
