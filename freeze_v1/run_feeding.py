from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core import NethraMemory, ResourceCloud
from feed_world import EnergyHandBallWorld
from world import BASE_COUNT, MOTOR_COUNT, HandBallWorld, support


def random_epoch(world, mem, active, rng, *, seed, steps, step0, sample_rate, persistence_floor):
    cloud = ResourceCloud(sample_rate, seed)
    prev = None
    for i in range(steps):
        for m in range(MOTOR_COUNT):
            if rng.random() < .02:
                if m in active:
                    active.remove(m)
                else:
                    active.add(m)
        obs = world.step(active)
        cur = support(obs)
        if prev is not None:
            cloud.observe(prev, cur, step0 + i + 1)
        prev = cur
    mem.checkpoint_from_cloud(cloud, step0 + steps, persistence_floor)
    return step0 + steps


def ensure_base_checkpoint(path: str, *, sample_rate: float, persistence_floor: float) -> dict:
    p = Path(path)
    if p.exists():
        mem, step, meta = NethraMemory.load(p)
        return {"path": str(p), "created": False, "step": step, "relations": len(mem.by_key), "metadata": meta}

    p.parent.mkdir(parents=True, exist_ok=True)
    mem = NethraMemory(BASE_COUNT, half_life=60000.0)
    world = HandBallWorld(balls=False, seed=18001)
    rng = random.Random(18001 ^ 0xA5A5)
    active = set()
    step = 0
    for e in range(8):
        step = random_epoch(world, mem, active, rng, seed=18001000 + e, steps=12000, step0=step,
                            sample_rate=sample_rate, persistence_floor=persistence_floor)
    world.enable_balls()
    for e in range(8):
        step = random_epoch(world, mem, active, rng, seed=18001008 + e, steps=12000, step0=step,
                            sample_rate=sample_rate, persistence_floor=persistence_floor)
    mem.save(p, step=step, metadata={"phase": "frozen_hand_ball", "seed": 18001})
    return {"path": str(p), "created": True, "step": step, "relations": len(mem.by_key)}


def feeding_lineage(args):
    (
        base_path, seed, resonance_enabled, steps, checkpoint_every,
        sample_rate, persistence_floor, execution_epsilon
    ) = args
    mem, base_step, _ = NethraMemory.load(base_path)
    world = EnergyHandBallWorld(balls=True, seed=seed)
    explore_rng = random.Random(seed ^ 0x13579BDF)
    field_rng = random.Random(seed ^ 0x2468ACE0)
    explore = set()
    prev = None
    cloud = ResourceCloud(sample_rate, seed ^ 0x55AA)
    step = base_step

    energy_sum = 0.0
    min_energy = world.energy
    source_contacts = 0
    contact_runs = []
    current_contact_run = 0
    low_steps = 0
    zero_steps = 0
    recoveries = 0
    was_low = False
    mean_active_motor_sum = 0
    resonance_motor_sum = 0.0
    first_contact = None
    checkpoints = []

    t0 = time.perf_counter()
    for i in range(steps):
        step += 1
        for m in range(MOTOR_COUNT):
            if explore_rng.random() < .02:
                if m in explore:
                    explore.remove(m)
                else:
                    explore.add(m)

        motors = set(explore)
        if resonance_enabled and prev is not None:
            scores = mem.resonance(prev, step)
            for m in range(MOTOR_COUNT):
                s = scores.get(m, 0.0)
                resonance_motor_sum += s
                if s > execution_epsilon:
                    p = 1.0 - math.exp(-s)
                    if field_rng.random() < p:
                        motors.add(m)

        obs = world.step(motors)
        cur = support(obs)
        if prev is not None:
            cloud.observe(prev, cur, step)
        prev = cur

        mean_active_motor_sum += len(motors)
        energy_sum += obs.energy
        min_energy = min(min_energy, obs.energy)
        if obs.energy < .25:
            low_steps += 1
            was_low = True
        if obs.energy <= 1e-12:
            zero_steps += 1
        if was_low and obs.energy > .55:
            recoveries += 1
            was_low = False

        if obs.source_contact:
            source_contacts += 1
            current_contact_run += 1
            if first_contact is None:
                first_contact = i + 1
        else:
            if current_contact_run:
                contact_runs.append(current_contact_run)
                current_contact_run = 0

        if (i + 1) % checkpoint_every == 0:
            cp = mem.checkpoint_from_cloud(cloud, step, persistence_floor)
            checkpoints.append(cp)
            cloud = ResourceCloud(sample_rate, seed ^ (0x55AA + i + 1))

    if current_contact_run:
        contact_runs.append(current_contact_run)
    if cloud.n:
        checkpoints.append(mem.checkpoint_from_cloud(cloud, step, persistence_floor))

    elapsed = time.perf_counter() - t0
    return {
        "seed": seed,
        "resonance": resonance_enabled,
        "steps": steps,
        "final_energy": world.energy,
        "mean_energy": energy_sum / steps,
        "min_energy": min_energy,
        "source_contacts": source_contacts,
        "contact_fraction": source_contacts / steps,
        "longest_contact_run": max(contact_runs, default=0),
        "first_contact": first_contact,
        "low_fraction": low_steps / steps,
        "zero_fraction": zero_steps / steps,
        "recoveries": recoveries,
        "mean_active_motors": mean_active_motor_sum / steps,
        "mean_motor_resonance_mass": resonance_motor_sum / (steps * MOTOR_COUNT),
        "persistent_relations": len(mem.by_key),
        "wall_s": elapsed,
        "us_per_step": 1e6 * elapsed / steps,
        "checkpoints": checkpoints,
    }


def summarize(rows):
    return {
        "n": len(rows),
        "final_energy_mean": statistics.mean(r["final_energy"] for r in rows),
        "mean_energy_mean": statistics.mean(r["mean_energy"] for r in rows),
        "contact_fraction_mean": statistics.mean(r["contact_fraction"] for r in rows),
        "longest_contact_run_median": statistics.median(r["longest_contact_run"] for r in rows),
        "low_fraction_mean": statistics.mean(r["low_fraction"] for r in rows),
        "zero_fraction_mean": statistics.mean(r["zero_fraction"] for r in rows),
        "recoveries_total": sum(r["recoveries"] for r in rows),
        "mean_active_motors": statistics.mean(r["mean_active_motors"] for r in rows),
        "us_per_step_mean": statistics.mean(r["us_per_step"] for r in rows),
        "survived_above_025": sum(r["final_energy"] > .25 for r in rows),
        "ended_above_055": sum(r["final_energy"] > .55 for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--lineages", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--checkpoint-every", type=int, default=12000)
    ap.add_argument("--sample-rate", type=float, default=.0625)
    ap.add_argument("--persistence-floor", type=float, default=.003)
    ap.add_argument("--execution-epsilon", type=float, default=.01)
    ap.add_argument("--out", default="feeding_results.json")
    args = ap.parse_args()

    base = ensure_base_checkpoint(args.base_checkpoint, sample_rate=args.sample_rate,
                                  persistence_floor=args.persistence_floor)
    seeds = [20001 + i for i in range(args.lineages)]
    jobs = []
    for enabled in (False, True):
        for seed in seeds:
            jobs.append((args.base_checkpoint, seed, enabled, args.steps, args.checkpoint_every,
                         args.sample_rate, args.persistence_floor, args.execution_epsilon))

    workers = max(1, min(args.workers, len(jobs), os.cpu_count() or 1))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(feeding_lineage, jobs))
    wall = time.perf_counter() - t0
    control = [r for r in rows if not r["resonance"]]
    resonant = [r for r in rows if r["resonance"]]
    report = {
        "base": base,
        "parameters": vars(args),
        "wall_s": wall,
        "control": summarize(control),
        "resonant": summarize(resonant),
        "rows": rows,
        "learner_semantics": {
            "reward": None,
            "energy_input": False,
            "food_label": False,
            "contact_input": False,
            "action_selector": False,
            "motor_bias": "generic symmetric persistent-relation resonance only",
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("base", "wall_s", "control", "resonant")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
