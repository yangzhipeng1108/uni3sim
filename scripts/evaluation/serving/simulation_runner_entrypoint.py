#!/usr/bin/env python3
"""Spawn-safe command-line entry point for the Serving simulator."""

from evaluation.serving.simulation.simulation_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
