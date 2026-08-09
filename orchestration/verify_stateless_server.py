#!/usr/bin/env python3
"""Verify that a listening stateless server is the expected model instance."""

from __future__ import annotations

import argparse
import json
import sys

import websockets.sync.client

from openpi_client import msgpack_numpy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()

    uri = f"ws://{args.host}:{args.port}"
    try:
        with websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            proxy=None,
            open_timeout=args.timeout,
            close_timeout=1,
        ) as connection:
            metadata = msgpack_numpy.unpackb(connection.recv(timeout=args.timeout))
    except Exception as exc:  # A readiness probe should report, not traceback.
        print(f"{uri}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    expected = {
        "protocol_version": 2,
        "serving_mode": "stateless-batched",
        "model_fingerprint": args.fingerprint,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        print(
            f"{uri}: server identity mismatch: {json.dumps(mismatches, sort_keys=True)}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                **expected,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
