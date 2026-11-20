#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Print the body of one SigV4-signed GET to a deployed profiler (service ``execute-api``).

    python3 sigv4_get.py https://gru.dev.profiler.neuron-profile.aws.dev/version sa-east-1

Credentials come from the botocore default chain, as in NKIRL's reward core. Exit status 1 on any
HTTP or credential error.
"""

from __future__ import annotations

import sys
import urllib.request

import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: sigv4_get.py <url> <region>", file=sys.stderr)
        return 2
    url, region = sys.argv[1], sys.argv[2]
    credentials = botocore.session.get_session().get_credentials()
    if credentials is None:
        print("no AWS credentials resolved", file=sys.stderr)
        return 1
    request = AWSRequest(method="GET", url=url)
    SigV4Auth(credentials, "execute-api", region).add_auth(request)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=dict(request.headers)), timeout=30) as resp:
            sys.stdout.write(resp.read().decode())
    except Exception as exc:
        print(f"GET {url}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
