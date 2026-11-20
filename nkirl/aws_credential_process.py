#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""A botocore ``credential_process`` that reads session credentials from a file of ``export`` lines.

WHY. NKIRL's reward core builds its SigV4 signer once per process. Credentials taken from
environment variables are then fixed for the life of the run, and a run that outlives the session
(about 12 hours) fails every profiler request. Credentials from a ``credential_process`` are
refreshable: this script reports an expiry ``REFRESH_S`` seconds ahead, and botocore runs it again
when less than 15 minutes remain. Replacing the file therefore takes effect within about 15 minutes,
with no restart. ``REFRESH_S`` must stay above botocore's 10-minute mandatory-refresh window, or
botocore runs this script on every signed request.

    # ~/.aws-style config file, named by AWS_CONFIG_FILE in every worker:
    [default]
    credential_process = python3 /path/to/aws_credential_process.py /path/to/creds.env

The file holds ``export AWS_ACCESS_KEY_ID=...``, ``export AWS_SECRET_ACCESS_KEY=...`` and
``export AWS_SESSION_TOKEN=...``. Nothing is logged.
"""

from __future__ import annotations

import datetime
import json
import sys

REFRESH_S = 1800


def read_exports(path: str) -> dict[str, str]:
    """Return the ``NAME=value`` pairs of the ``export`` lines in ``path``."""
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("export "):
                name, _, value = line[len("export ") :].partition("=")
                values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aws_credential_process.py <file of export lines>", file=sys.stderr)
        return 2
    values = read_exports(sys.argv[1])
    missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not values.get(k)]
    if missing:
        print(f"credentials file lacks {missing}", file=sys.stderr)
        return 1
    expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=REFRESH_S)
    out = {
        "Version": 1,
        "AccessKeyId": values["AWS_ACCESS_KEY_ID"],
        "SecretAccessKey": values["AWS_SECRET_ACCESS_KEY"],
        "Expiration": expiry.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if values.get("AWS_SESSION_TOKEN"):
        out["SessionToken"] = values["AWS_SESSION_TOKEN"]
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
