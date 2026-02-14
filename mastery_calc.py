#!/usr/bin/env python3

import argparse
import bisect
import copy
import dataclasses
import math
from typing import Iterator, Self

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sciform


# Global constants

# fmt: off

# Base game data

GAME_SPEED = 5.0 * 0.8
WAVE_DURATION = 26.0
WAVE_COOLDOWN = 5.0

TIERS = list(range(1, 19))
TIER_COIN_BONUS = [1.0, 1.8, 2.6, 3.4, 4.2, 5.0, 5.8, 6.6, 7.5, 8.7, 10.3, 12.2, 14.7, 17.6, 21.3, 25.2, 29.1, 33.0]
assert len(TIERS) == len(TIER_COIN_BONUS)
TIER_CELL_DROP_MIN = [*([1] * 13), 7, 11, 12, 12, 12]
assert len(TIERS) == len(TIER_CELL_DROP_MIN)
TIER_CELL_DROP_MAX = [*range(1, 14), 14, 15, 16, 17, 18]
assert len(TIERS) == len(TIER_CELL_DROP_MAX)
TIER_REROLL_DROP = [1, 2, 3, 4, 6, 8, 12, 18, 25, 32, 40, 45, 50, 55, 60, 65, 70, 75]
assert len(TIERS) == len(TIER_REROLL_DROP)
TIER_BOSS_PERIOD = [*([10] * 13), 9, 8, 7, 6, 5]
assert len(TIERS) == len(TIER_BOSS_PERIOD)
SPAWN_RATE_SEQUENCE = [10, 11, 13, 15, 17, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 37, 39, 40, 42, 44, 46, 48, 49, 50, 52, 54, 56]
SPAWN_RATE_WAVES = [1, 3, 6, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]
assert len(SPAWN_RATE_SEQUENCE) == len(SPAWN_RATE_WAVES)
SPAWN_RATE_FACTOR = 8 * 1.9 / 100 # 8 times per second, +90% from EB, convert to percent

SPAWN_CHANCE_TABLE = {
    "fast": [0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.10, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13, 0.13, 0.14, 0.15, 0.17, 0.18, 0.19, 0.20, 0.21, 0.21, 0.22, 0.23, 0.24, 0.24],
    "tank": [0.00, 0.02, 0.04, 0.06, 0.07, 0.08, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.13, 0.14, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.19, 0.20, 0.20, 0.20, 0.21, 0.21, 0.22],
    "ranged": [0.00, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.09, 0.10, 0.11, 0.11, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.19, 0.19, 0.20, 0.21],
    "protector": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
}
assert all(len(row) == len(SPAWN_RATE_SEQUENCE) for row in SPAWN_CHANCE_TABLE.values())
SPAWN_CHANCE_TABLE["basic"] = [
    1.0 - sum(row[i] for row in SPAWN_CHANCE_TABLE.values())
    for i in range(len(SPAWN_RATE_SEQUENCE))
]

COIN_DROP_TABLE = {
    "basic": 0.33,
    "fast": 2.0,
    "tank": 4.0,
    "ranged": 2.0,
    "protector": 3.0,
}
assert sorted(COIN_DROP_TABLE.keys()) == sorted(SPAWN_CHANCE_TABLE.keys())
COIN_DROP_TABLE["scatter"] = 4.0
COIN_DROP_TABLE["vampire"] = 4.0
COIN_DROP_TABLE["ray"] = 4.0
COIN_DROP_TABLE["boss"] = 0.0
COIN_DROP_TABLE["saboteur"] = 0.0
COIN_DROP_TABLE["commander"] = 0.0
COIN_DROP_TABLE["overcharge"] = 0.0

ELITE_SPAWN_CHANCE_TABLE = [0.00, 0.01, 0.04, 0.09, 0.15, 0.24, 0.36, 0.48, 0.63, 0.81, 1.00]
ELITE_SINGLE_SPAWN_WAVES_TABLE = [
    [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    [0, 450, 900, 1350, 1800, 2700, 3600, 4500, 5400, 6300, 7200],
    [0, 405, 810, 1215, 1620, 2430, 3240, 4050, 4860, 5670, 6480],
    [0, 365, 729, 1094, 1458, 2187, 2916, 3645, 4374, 5103, 5832],
    [0, 328, 656, 984, 1312, 1968, 2624, 3281, 3937, 4593, 5249],
    [0, 295, 590, 886, 1181, 1771, 2362, 2952, 3543, 4133, 4724],
    [0, 266, 531, 797, 1063, 1594, 2126, 2657, 3189, 3720, 4252],
    [0, 239, 478, 717, 957, 1435, 1913, 2391, 2870, 3348, 3826],
    [0, 215, 430, 646, 861, 1291, 1722, 2152, 2583, 3013, 3444],
    [0, 194, 387, 581, 775, 1162, 1550, 1937, 2325, 2712, 3099],
    [0, 174, 349, 523, 697, 1046, 1395, 1743, 2092, 2441, 2789],
    [0, 157, 314, 471, 628, 941, 1255, 1569, 1883, 2197, 2510],
    [0, 141, 282, 424, 565, 847, 1130, 1412, 1695, 1977, 2259],
    [0, 127, 254, 381, 508, 763, 1017, 1271, 1525, 1779, 2033],
    [0, 114, 229, 343, 458, 686, 915, 1144, 1373, 1601, 1830],
    [0, 41, 102, 205, 308, 411, 617, 823, 1029, 1235, 1441],
    [0, 37, 92, 185, 277, 370, 555, 741, 926, 1111, 1297],
    [0, 33, 83, 166, 250, 333, 500, 667, 833, 1000, 1167],
]
assert all(len(row) == len(ELITE_SPAWN_CHANCE_TABLE) for row in ELITE_SINGLE_SPAWN_WAVES_TABLE)
ELITE_DOUBLE_SPAWN_WAVES_TABLE = [
    [0, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000],
    [0, 7200, 8100, 9000, 9900, 10800, 11700, 12600, 13500, 14400, 15300],
    [0, 6480, 7290, 8100, 8910, 9720, 10530, 11340, 12150, 12960, 13770],
    [0, 5832, 6561, 7290, 8019, 8748, 9477, 10206, 10935, 11664, 12393],
    [0, 5249, 5905, 6561, 7217, 7873, 8529, 9185, 9841, 10497, 11153],
    [0, 4724, 5314, 5905, 6495, 7086, 7676, 8267, 8857, 9448, 10038],
    [0, 4252, 4783, 5314, 5846, 6377, 6909, 7440, 7972, 8503, 9034],
    [0, 3826, 4305, 4783, 5261, 5740, 6218, 6696, 7174, 7653, 8131],
    [0, 3444, 3874, 4305, 4735, 5166, 5596, 6027, 6457, 6887, 7318],
    [0, 3099, 3487, 3874, 4262, 4649, 5036, 5424, 5811, 6199, 6586],
    [0, 2789, 3138, 3487, 3835, 4184, 4533, 4881, 5230, 5579, 5928],
    [0, 2510, 2824, 3138, 3452, 3766, 4080, 4394, 4708, 5022, 5336],
    [0, 2259, 2542, 2824, 3107, 3389, 3672, 3954, 4236, 4519, 4801],
    [0, 2033, 2288, 2542, 2796, 3050, 3304, 3559, 3813, 4067, 4321],
    [0, 1830, 2059, 2288, 2516, 2745, 2974, 3203, 3432, 3660, 3889],
    [0, 1441, 1647, 1853, 2058, 2264, 2470, 2676, 2882, 3088, 3294],
    [0, 1297, 1482, 1667, 1853, 2038, 2223, 2408, 2594, 2779, 2964],
    [0, 1167, 1334, 1500, 1667, 1834, 2001, 2168, 2334, 2501, 2668],
]
assert all(len(row) == len(ELITE_SPAWN_CHANCE_TABLE) for row in ELITE_DOUBLE_SPAWN_WAVES_TABLE)
assert len(ELITE_SINGLE_SPAWN_WAVES_TABLE) == len(ELITE_DOUBLE_SPAWN_WAVES_TABLE)

FLEET_MIN_WAVE_SPAWN_TABLE = [*([None] * 13), 2495, 1495, 995, 495, 95, 45, 5, 5]
FLEET_SPAWN_PERIOD_WAVE_TABLE = [*([None] * 13), 1000, 750, 500, 250, 100, 50, 10, 10]
FLEET_SPAWN_COUNT_TABLE = [*([0] * 13), 1, 1, 1, 1, 1, 1, 1, 2]
FLEET_REROLL_SHARD_DROP_TABLE = [*([0] * 13), 1080, 1200, 1350, 1500, 1650, 1800, 1950, 2100]
assert len(FLEET_MIN_WAVE_SPAWN_TABLE) == len(FLEET_SPAWN_PERIOD_WAVE_TABLE)
assert len(FLEET_MIN_WAVE_SPAWN_TABLE) == len(FLEET_SPAWN_COUNT_TABLE)
assert len(FLEET_MIN_WAVE_SPAWN_TABLE) == len(FLEET_REROLL_SHARD_DROP_TABLE)

FLEET_MODULE_SHARD_DROP_TABLE = [
    (1, 5), (250, 6), (500, 7), (750, 8),
    (1000, 9), (1250, 10), (1500, 11), (1750, 12),
    (2000, 13), (2250, 14), (2500, 15), (2750, 16),
    (3000, 17), (3250, 18), (3500, 19), (3750, 20),
    (4000, 21), (4250, 22), (4500, 23), (4750, 24),
    (5000, 25),
]

FLEET_REROLL_SHARD_DROP_CHANCE = 0.8
FLEET_MODULE_SHARD_DROP_CHANCE = 0.2
BOSS_REROLL_SHARD_DROP_CHANCE = 0.15
BOSS_COMMON_MODULE_DROP_CHANCE = 0.03
COMMON_MODULE_VALUE = 10
RARE_MODULE_DROP_CHANCE = 0.015
RARE_MODULE_VALUE = 30
RECOVERY_PACKAGE_CHANCE = 82 # %

# Perk data

STANDARD_PERK_CHANCE = 0.65
ULTIMATE_PERK_CHANCE = 0.20
TRADEOFF_PERK_CHANCE = 0.15
assert STANDARD_PERK_CHANCE + ULTIMATE_PERK_CHANCE + TRADEOFF_PERK_CHANCE == 1.0

# Quantity of each standard perk
STANDARD_PERKS = {
    "std-health": 5,
    "std-damage": 5,
    "std-coin-bonus": 5,
    "std-defabs": 5,
    "std-cash-bonus": 5,
    "std-regen": 5,
    "std-interest": 5,
    "std-lm-damage": 5,
    "std-freeup-chance": 5,
    "std-def%": 5,
    "std-bounce-shot": 3,
    "std-pwr": 3,
    "std-orbs": 2,
    "std-random-uw": 1,
    "std-game-speed": 1,
}

# Quantity of each UW perk
ULTIMATE_PERKS = {
    "uw-sm": 1,
    "uw-ps": 1,
    "uw-dw": 1,
    "uw-ilm": 1,
    "uw-gt": 1,
    "uw-cl": 1,
    "uw-cf": 1,
    "uw-bh": 1,
    "uw-sl": 1,
}

# Quantity of each TO perk
TRADEOFF_PERKS = {
    "to-tower-damage": 1,
    "to-coin": 1,
    "to-enemy-health": 1,
    "to-enemy-damage": 1,
    "to-enemy-range": 1,
    "to-enemy-speed": 1,
    "to-cash": 1,
    "to-regen": 1,
    "to-boss-health": 1,
    "to-lifesteal": 1,
}

ALL_PERKS = {**STANDARD_PERKS, **ULTIMATE_PERKS, **TRADEOFF_PERKS}

# Bonuses for each perk
PERK_BONUSES = {
    "std-coin-bonus": 1.15,
    "std-pwr": -0.2,
    "std-game-speed": 1.0,
    "std-freeup-chance": 0.05,
    "uw-gt": 1.5,
    "to-coin": 1.8,
}

PERK_PRIORITY_ORDER = [
    "std-pwr",
    "std-game-speed",
    "to-coin",
    "uw-gt",
    "std-coin-bonus",
    "std-freeup-chance",
]

FIRST_PERK_CHOICE = "std-pwr"

PERK_BANS = [
    "to-tower-damage",
    "to-enemy-health",
    "to-enemy-range",
    "to-enemy-speed",
    "to-cash",
    "to-boss-health",
    "std-defabs",
    "std-interest",
]


# Card data

WAVE_SKIP_CHANCE = 0.19
WAVE_SKIP_BONUS = 1.10

# Mastery data

CASH_MASTERY_TABLE = [0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.032, 0.036, 0.040]
COIN_MASTERY_TABLE = [1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30]
CRITICAL_COIN_MASTERY_TABLE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ENEMY_BALANCE_MASTERY_TABLE = [0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60]
EXTRA_ORB_MASTERY_TABLE = [1.04, 1.08, 1.12, 1.16, 1.20, 1.24, 1.28, 1.32, 1.36, 1.40]
INTRO_SPRINT_MASTERY_TABLE = [180, 360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800]
RECOVERY_PACKAGE_CHANCE_MASTERY_TABLE = [0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.032, 0.036, 0.040]
WAVE_ACCELERATOR_MASTERY_TABLE = [
    [1, 3, 5, 18, 36, 55, 73, 91, 136, 182, 227, 273, 364, 545, 727, 909, 1364, 1818, 2273, 2727, 3182, 3636, 4091, 4545, 5000, 5455, 5909],
    [1, 3, 5, 17, 33, 50, 67, 83, 125, 167, 208, 250, 333, 500, 667, 833, 1250, 1667, 2083, 2500, 2917, 3333, 3750, 4167, 4583, 5000, 5417],
    [1, 2, 5, 15, 31, 46, 62, 77, 115, 154, 192, 231, 308, 462, 615, 769, 1154, 1538, 1923, 2308, 2692, 3077, 3462, 3846, 4231, 4615, 5000],
    [1, 2, 4, 14, 29, 43, 57, 71, 107, 143, 179, 214, 286, 429, 571, 714, 1071, 1429, 1786, 2143, 2500, 2857, 3214, 3571, 3929, 4286, 4643],
    [1, 2, 4, 13, 27, 40, 53, 67, 100, 133, 167, 200, 267, 400, 533, 667, 1000, 1333, 1667, 2000, 2333, 2667, 3000, 3333, 3667, 4000, 4333],
    [1, 2, 4, 13, 25, 38, 50, 63, 94, 125, 156, 188, 250, 375, 500, 625, 938, 1250, 1563, 1875, 2188, 2500, 2813, 3125, 3438, 3750, 4063],
    [1, 2, 4, 12, 24, 35, 47, 59, 88, 118, 147, 176, 235, 353, 471, 588, 882, 1176, 1471, 1765, 2059, 2353, 2647, 2941, 3235, 3529, 3824],
    [1, 2, 3, 11, 22, 33, 44, 56, 83, 111, 139, 167, 222, 333, 444, 556, 833, 1111, 1389, 1667, 1944, 2222, 2500, 2778, 3056, 3333, 3611],
    [1, 2, 3, 11, 21, 32, 42, 53, 79, 105, 132, 158, 211, 316, 421, 526, 789, 1053, 1316, 1579, 1842, 2105, 2368, 2632, 2895, 3158, 3421],
    [1, 2, 3, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250]
]
assert all(len(row) == len(SPAWN_RATE_SEQUENCE) for row in WAVE_ACCELERATOR_MASTERY_TABLE)
WAVE_SKIP_MASTERY_TABLE = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

MASTERY_LEVELS = [None, *range(0, 10)]
MASTERY_LEVEL_NAMES = ["locked"] + [str(x) for x in range(0, 10)]
MASTERY_DISPLAY_NAMES = {
    "cash": "Cash",
    "coin": "Coin",
    "critical-coin": "CritCoin",
    "enemy-balance": "EB",
    "extra-orb": "EO",
    "intro-sprint": "IS",
    "recovery-package": "RPC",
    "wave-accelerator": "WA",
    "wave-skip": "WS",
}
MASTERY_STONE_COSTS = {
    "cash": 500,
    "coin": 1250,
    "critical-coin": 1000,
    "enemy-balance": 1000,
    "extra-orb": 750,
    "intro-sprint": 1250,
    "recovery-package": 1000,
    "wave-accelerator": 1000,
    "wave-skip": 1000,
}

REWARD_NAMES = ["coins", "cells", "rerolls", "modules"]
# fmt: on


# Data types


@dataclasses.dataclass
class Simulation:
    name: str = ""
    mastery: str | None = None
    level: int | None = None
    skip: bool = False

    # Estimates / inputs
    tier: int = 1
    max_waves: int = 0
    orb_hits: float = 1.0
    reward: str = "coins"
    sum_total_stone_cost: bool = False
    bhd_bonus: float = 0
    golden_combo: float = 0

    # Workshop stats
    free_upgrade_chances: dict[str, float] = dataclasses.field(default_factory=dict)
    package_chance: float = RECOVERY_PACKAGE_CHANCE / 100
    enemy_level_skip_chances: dict[str, float] = dataclasses.field(default_factory=dict)

    # Lab levels
    standard_perk_bonus_lab: int = 25
    improved_tradeoff_perk_lab: int = 10
    perk_option_quantity_lab: int = 2
    perk_waves_required_lab: int = 25
    first_perk_choice: str = FIRST_PERK_CHOICE
    perk_priority_order: list[str] = dataclasses.field(default_factory=lambda: PERK_PRIORITY_ORDER)
    perk_bans: list[str] = dataclasses.field(default_factory=lambda: PERK_BANS)

    # Mastery levels
    cash: int | None = None
    coin: int | None = None
    critical_coin: int | None = None
    enemy_balance: int | None = None
    extra_orb: int | None = None
    intro_sprint: int | None = None
    recovery_package: int | None = None
    wave_accelerator: int | None = None
    wave_skip: int | None = None

    def stone_cost(self) -> int:
        if not self.sum_total_stone_cost:
            return 0 if self.mastery is None else MASTERY_STONE_COSTS[self.mastery]
        stone_cost = 0
        if self.cash is not None:
            stone_cost += MASTERY_STONE_COSTS["cash"]
        if self.coin is not None:
            stone_cost += MASTERY_STONE_COSTS["coin"]
        if self.critical_coin is not None:
            stone_cost += MASTERY_STONE_COSTS["critical-coin"]
        if self.enemy_balance is not None:
            stone_cost += MASTERY_STONE_COSTS["enemy-balance"]
        if self.extra_orb is not None:
            stone_cost += MASTERY_STONE_COSTS["extra-orb"]
        if self.intro_sprint is not None:
            stone_cost += MASTERY_STONE_COSTS["intro-sprint"]
        if self.recovery_package is not None:
            stone_cost += MASTERY_STONE_COSTS["recovery-package"]
        if self.wave_accelerator is not None:
            stone_cost += MASTERY_STONE_COSTS["wave-accelerator"]
        if self.wave_skip is not None:
            stone_cost += MASTERY_STONE_COSTS["wave-skip"]
        return stone_cost

    def standard_perk_bonus(self) -> float:
        return 1 + self.standard_perk_bonus_lab / 100

    def tradeoff_perk_bonus(self) -> float:
        return 1 + self.improved_tradeoff_perk_lab / 100

    def perk_option_quantity(self) -> int:
        return 2 + self.perk_option_quantity_lab

    def perk_waves_required(self, base: int) -> int:
        return base - self.perk_waves_required_lab


def perks_default_factory() -> dict[str, float]:
    return {
        name: 0.0 for name in ALL_PERKS.keys()
    }


@dataclasses.dataclass
class Perks:
    perks: dict[str, float] = dataclasses.field(default_factory=perks_default_factory)

    def pwr_bonus(self, sim: Simulation) -> float:
        return 1.0 + self.perks.get("std-pwr", 0) * PERK_BONUSES["std-pwr"] * sim.standard_perk_bonus()

    def game_speed_factor(self, sim: Simulation) -> float:
        return (GAME_SPEED + self.perks.get("std-game-speed", 0) * PERK_BONUSES["std-game-speed"] * sim.standard_perk_bonus()) / GAME_SPEED

    def coin_bonus(self, sim: Simulation) -> float:
        bonus = 1.0
        bonus *= self.perks.get("std-coin-bonus", 0) * PERK_BONUSES["std-coin-bonus"] * sim.standard_perk_bonus()
        bonus *= self.perks.get("uw-gt", 0) * PERK_BONUSES["uw-gt"]
        bonus *= self.perks.get("to-coin", 0) * PERK_BONUSES["to-coin"] * sim.tradeoff_perk_bonus()
        return 1.0 + bonus

    def free_upgrade_chance_bonus(self, sim: Simulation) -> float:
        return self.perks.get("std-freeup-chance", 0) * PERK_BONUSES["std-freeup-chance"] * sim.standard_perk_bonus()


def free_upgrades_default_factory() -> dict[str, float]:
    return {
        "attack": 0.0,
        "defense": 0.0,
        "utility": 0.0,
    }


def enemy_level_skip_default_factory() -> dict[str, float]:
    return {
        "attack": 0.0,
        "health": 0.0,
    }


def enemies_default_factory() -> dict[str, float]:
    return {name: 0.0 for name in COIN_DROP_TABLE.keys()}


@dataclasses.dataclass
class Events:
    wave: int = 0
    wave_skip: float = 0.0
    free_upgrades: dict[str, float] = dataclasses.field(
        default_factory=free_upgrades_default_factory
    )
    recovery_packages: float = 0.0
    enemy_level_skips: dict[str, float] = dataclasses.field(
        default_factory=enemy_level_skip_default_factory
    )
    enemies: dict[str, float] = dataclasses.field(
        default_factory=enemies_default_factory
    )

    def elite_enemy_count(self) -> float:
        return self.enemies["scatter"] + self.enemies["vampire"] + self.enemies["ray"]

    def fleet_enemy_count(self) -> float:
        return self.enemies["saboteur"] + self.enemies["commander"] + self.enemies["overcharge"]

    def scatter_children_count(self) -> float:
        # Each scatter splits in half 4 times.
        return self.enemies["scatter"] * sum(1 << i for i in range(1, 5))

    def total_enemy_count(self) -> float:
        return sum(self.enemies.values()) + self.scatter_children_count()

    def __iadd__(self, other: Self) -> Self:
        self.wave_skip += other.wave_skip
        for key, value in other.free_upgrades.items():
            self.free_upgrades[key] += value
        self.recovery_packages += other.recovery_packages
        for key, value in other.enemy_level_skips.items():
            self.enemy_level_skips[key] += value
        for key, value in other.enemies.items():
            self.enemies[key] += value
        return self

    def __add__(self, other: Self) -> Self:
        result = copy.deepcopy(self)
        result += other
        return result


@dataclasses.dataclass
class Rewards:
    coins: float = 0.0
    elite_cells: float = 0.0
    reroll_shards: float = 0.0
    module_shards: float = 0.0

    def __iadd__(self, other: Self) -> Self:
        self.coins += other.coins
        self.elite_cells += other.elite_cells
        self.reroll_shards += other.reroll_shards
        self.module_shards += other.module_shards
        return self

    def __add__(self, other: Self) -> Self:
        result = copy.deepcopy(self)
        result += other
        return result

    def __isub__(self, other: Self) -> Self:
        self.coins -= other.coins
        self.elite_cells -= other.elite_cells
        self.reroll_shards -= other.reroll_shards
        self.module_shards -= other.module_shards
        return self

    def __sub__(self, other: Self) -> Self:
        result = copy.deepcopy(self)
        result -= other
        return result

    def __imul__(self, factor: float | int) -> Self:
        self.coins *= factor
        self.elite_cells *= factor
        self.reroll_shards *= factor
        self.module_shards *= factor
        return self

    def __mul__(self, factor: float | int) -> Self:
        result = copy.deepcopy(self)
        result *= factor
        return result


@dataclasses.dataclass
class SimulationWaveResult:
    wave: int
    elapsed_time: float
    cumulative_events: Events
    cumulative_rewards: Rewards


@dataclasses.dataclass
class SimulationRunResult:
    wave_results: list[SimulationWaveResult]
    total: float | None = None
    relative: float | None = None
    roi: float | None = None


@dataclasses.dataclass
class PlotLine:
    name: str
    mastery: str | None = None
    xs: list[float] = dataclasses.field(default_factory=list)
    ys: list[float] = dataclasses.field(default_factory=list)
    relative: float | None = None
    roi: float | None = None


@dataclasses.dataclass
class Plot:
    title: str
    xlabel: str
    ylabel: str
    top: float | None = None
    bottom: float | None = None
    lines: list[PlotLine | None] = dataclasses.field(default_factory=list)


# Argument handling


def tier_and_wave_arg(arg: str) -> tuple[int, int]:
    tier, _, wave = arg.partition(":")
    return int(tier), int(wave)


def add_common_args(parser: argparse.ArgumentParser):
    # Simulation events
    parser.add_argument(
        "--tier",
        type=int,
        choices=TIERS,
        default=1,
        help="Tier to simulate",
    )
    parser.add_argument(
        "--orb-hits",
        type=float,
        default=1.0,
        help="Average portion of enemies hit by orbs [0.0-1.0]",
    )
    parser.add_argument(
        "--freeup-chance",
        nargs=3,
        type=int,
        default=[75, 75, 75],
        help="Free-upgrade chances (attack %%, defense %%, utility %%)",
    )
    parser.add_argument(
        "--package-chance",
        type=int,
        default=RECOVERY_PACKAGE_CHANCE,
        help="Free-upgrade chances (attack %%, defense %%, utility %%)",
    )
    parser.add_argument(
        "--bhd",
        choices=[0, 3, 5, 7, 10],
        type=int,
        default=0,
        help="BHD free-upgrade coin multiplier (%%)",
    )
    parser.add_argument(
        "--golden-combo",
        type=float,
        default=0,
        help="Golden Combo multiplier (%%)",
    )

    # Reward normalization
    parser.add_argument(
        "--reward",
        type=str,
        default="coins",
        choices=REWARD_NAMES,
        help="Which reward to plot and compare",
    )
    parser.add_argument(
        "--difference",
        "-d",
        action="store_true",
        default=False,
        help="Subtract the baseline configuration from all results",
    )
    parser.add_argument(
        "--elapsed",
        action="store_true",
        default=False,
        help="Normalize results by elapsed time",
    )
    parser.add_argument(
        "--relative",
        "-r",
        action="store_true",
        default=False,
        help="Normalize all results against the baseline configuration",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Normalize all results against mastery stone cost",
    )
    parser.add_argument(
        "--sum-total-stone-cost",
        action="store_true",
        default=False,
        help="Consider the stone cost of all masteries when normalizing results",
    )

    # Output options
    parser.add_argument(
        "--truncate",
        action="store_true",
        default=False,
        help="Truncate runs to the end of the shortest simulation",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        default=False,
        help="Extend shorter runs with their average results",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        default=False,
        help="Crop results of the plot vertically",
    )
    parser.add_argument(
        "--no-print",
        action="store_false",
        default=True,
        dest="print",
        help="Do not print results",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        default=True,
        dest="plot",
        help="Do not plot results",
    )
    parser.add_argument("--output", "-o", default=None, help="Filename for saved plot")

    # Masteries
    parser.add_argument(
        "--cash",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Cash mastery level",
    )
    parser.add_argument(
        "--coin",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Coin mastery level",
    )
    parser.add_argument(
        "--critical-coin",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Critical coin mastery level",
    )
    parser.add_argument(
        "--enemy-balance",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Enemy balance mastery level",
    )
    parser.add_argument(
        "--extra-orb",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Extra orb mastery level",
    )
    parser.add_argument(
        "--recovery-package",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Recovery package mastery level",
    )
    parser.add_argument(
        "--intro-sprint",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Intro sprint mastery level",
    )
    parser.add_argument(
        "--wave-accelerator",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Wave accelerator mastery level",
    )
    parser.add_argument(
        "--wave-skip",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="Wave skip mastery level",
    )
    parser.add_argument(
        "--rerolls-with-cash",
        choices=MASTERY_LEVEL_NAMES,
        default=None,
        help="The sim changing reroll-affecting masteries (EB#, IS#, WS#) should set cash to this level",
    )


def mastery_level(name: str | None) -> int | None:
    if name is None or name == "locked":
        return None
    if int(name) in range(0, 10):
        return int(name)
    raise ValueError(f"Invalid mastery level: {name}")


def convert_mastery_args(args: argparse.Namespace) -> None:
    args.cash = mastery_level(args.cash)
    args.coin = mastery_level(args.coin)
    args.critical_coin = mastery_level(args.critical_coin)
    args.enemy_balance = mastery_level(args.enemy_balance)
    args.extra_orb = mastery_level(args.extra_orb)
    args.intro_sprint = mastery_level(args.intro_sprint)
    args.recovery_package = mastery_level(args.recovery_package)
    args.wave_accelerator = mastery_level(args.wave_accelerator)
    args.wave_skip = mastery_level(args.wave_skip)
    args.rerolls_with_cash = mastery_level(args.rerolls_with_cash)


def common_args_description(args: argparse.Namespace, baseline_name: str) -> list[str]:
    desc = []

    # Simulation events
    if args.tier is not None:
        assert args.wave is not None
        desc.append(f"T{args.tier}W{args.wave}")
    if args.orb_hits != 1.0:
        desc.append(f"orbs {args.orb_hits:.2%}")
    if args.bhd > 0:
        desc.append(f"bhd {args.bhd}%")
    if args.golden_combo > 0:
        desc.append(f"GT+ {args.golden_combo}%")

    # Reward normalization
    desc.append(args.reward)
    if args.elapsed:
        desc.append("per hour")
    if args.difference:
        desc.append(f"minus {baseline_name}")
    if args.relative:
        desc.append(f"over {baseline_name}")
    if args.roi:
        desc.append("per stone")

    # Output options
    if args.truncate:
        desc.append("truncated")
    if args.crop:
        desc.append("cropped")

    # Masteries
    if args.cash is not None:
        desc.append(f"{MASTERY_DISPLAY_NAMES['cash']}#{args.cash}")
    if args.coin is not None:
        desc.append(f"{MASTERY_DISPLAY_NAMES['coin']}#{args.coin}")
    if args.critical_coin is not None:
        desc.append(f"{MASTERY_DISPLAY_NAMES['critical-coin']}#{args.critical_coin}")
    if args.enemy_balance is not None:
        desc.append(f"{MASTERY_DISPLAY_NAMES['enemy-balance']}#{args.enemy_balance}")
    if args.extra_orb is not None:
        desc.append(f"{MASTERY_DISPLAY_NAMES['extra-orb']}#{args.extra_orb}")
    if args.intro_sprint is not None:
        desc.append(f"{MASTERY_DISPLAY_NAMES['intro-sprint']}#{args.intro_sprint}")
    if args.recovery_package is not None:
        desc.append(
            f"{MASTERY_DISPLAY_NAMES['recovery-package']}#{args.recovery_package}"
        )
    if args.wave_accelerator is not None:
        desc.append(
            f"{MASTERY_DISPLAY_NAMES['wave-accelerator']}#{args.wave_accelerator}"
        )
    if args.wave_skip is not None:
        desc.append(f"{MASTERY_DISPLAY_NAMES['wave-skip']}#{args.wave_skip}")
    if args.rerolls_with_cash is not None:
        desc.append(
            f"rerolls with {MASTERY_DISPLAY_NAMES['cash']}#{args.rerolls_with_cash}"
        )

    return desc


# Simulation logic


def max_intro_wave(sim: Simulation) -> int:
    return (
        100
        if sim.intro_sprint is None
        else INTRO_SPRINT_MASTERY_TABLE[sim.intro_sprint]
    )


def perk_count_at_wave(
    sim: Simulation,
    perks: Perks,
    wave: int,
) -> float:
    pwr_bonus = perks.pwr_bonus(sim)
    pwr_waves = [
        (20, sim.perk_waves_required(200) * pwr_bonus),
        (20, sim.perk_waves_required(250) * pwr_bonus),
        (10, sim.perk_waves_required(300) * pwr_bonus),
    ]
    last_wave = 0
    perk_count = 0
    for perk_quantity, waves_per_perk in pwr_waves:
        max_wave = last_wave + perk_quantity * waves_per_perk
        if wave < max_wave:
            return perk_count + ((wave - last_wave) / waves_per_perk)
        last_wave = max_wave
        perk_count += perk_quantity
    return perk_count


def perks_confidence_default_factory() -> dict[str, list[float]]:
    return {
        perk: [0.0] * qty
        for perk, qty in ALL_PERKS.items()
    }


@dataclasses.dataclass
class PerksConfidence:
    """
    Confidence that each perk has been selected a given number of times.
    Factors in first-perk, bans, and priority.
    """

    count: int = 0

    # For each perk, the confidence that it has been selected 1..n times.
    # n is the maximum quantity of the perk.
    perks: dict[str, list[float]] = dataclasses.field(default_factory=perks_confidence_default_factory)

    def __iadd__(self, other: Self) -> Self:
        for perk in self.perks.keys():
            for i in range(0, len(self.perks[perk])):
                self.perks[perk][i] += other.perks[perk][i]
        return self

    def __add__(self, other: Self) -> Self:
        result = copy.deepcopy(self)
        result += other
        return result

    def __imul__(self, factor: float | int) -> Self:
        for perk in self.perks.keys():
            for i in range(0, len(self.perks[perk])):
                self.perks[perk][i] *= factor
        return self

    def __mul__(self, factor: float | int) -> Self:
        result = copy.deepcopy(self)
        result *= factor
        return result

    def reduce(self) -> Perks:
        return Perks(perks={perk: sum(confidences) for perk, confidences in self.perks.items()})


def perk_options_default_factory() -> dict[str, float]:
    return {
        perk: 0.0
        for perk in ALL_PERKS.keys()
    }


@dataclasses.dataclass
class PerkOptions:
    options: dict[str, float] = dataclasses.field(default_factory=perk_options_default_factory)

    def __iadd__(self, other: Self) -> Self:
        for perk in self.options.keys():
            self.options[perk] += other.options[perk]
        return self

    def __add__(self, other: Self) -> Self:
        result = copy.deepcopy(self)
        result += other
        return result

    def __imul__(self, factor: float | int) -> Self:
        for perk in self.options.keys():
            self.options[perk] *= factor
        return self

    def __mul__(self, factor: float | int) -> Self:
        result = copy.deepcopy(self)
        result *= factor
        return result

    def inorm(self) -> Self:
        magnitude = sum(self.options.values())
        if magnitude:
            for perk in self.options.keys():
                self.options[perk] /= magnitude
        return self

    def norm(self) -> Self:
        result = copy.deepcopy(self)
        result.inorm()
        return result


@dataclasses.dataclass
class PerkWaveEstimator:
    confidences: list[PerksConfidence]

    def lower(self, perk_count: float) -> PerksConfidence:
        if perk_count < 1.0:
            return PerksConfidence()
        return self.confidences[math.floor(perk_count - 1.0)]

    def higher(self, perk_count: float) -> PerksConfidence:
        if perk_count >= len(self.confidences):
            return self.confidences[-1]
        return self.confidences[math.floor(perk_count)]

    def average(self, perk_count: float) -> PerksConfidence:
        lower_confidence = self.lower(perk_count)
        upper_confidence = self.higher(perk_count)
        lambda_ = perk_count % 1.0
        return lower_confidence + (upper_confidence + lower_confidence * -1.0) * lambda_

    def estimate(self, sim: Simulation, wave: int, perks: Perks) -> Perks:
        perk_count = sum(perks.perks.values())
        next_perk_count = perk_count_at_wave(sim, perks, wave)
        while perk_count < next_perk_count:
            perk_count = next_perk_count
            perks = self.average(perk_count).reduce()
            next_perk_count = perk_count_at_wave(sim, perks, wave)
        return perks


def perk_category_option_chances(sim: Simulation, options: PerkOptions, confidence: PerksConfidence, category: dict[str, int], factor: float) -> PerkOptions:
    next_options = PerkOptions()
    for perk in category.keys():
        if perk not in sim.perk_bans:
            # Perks can't appear if they are already in the option set, or if their
            # quantity has been fully exhausted.
            chance = (1.0 - options.options[perk])
            chance *= (1.0 - confidence.perks[perk][-1])
            next_options.options[perk] = chance
    return next_options.inorm() * factor


def perk_option_chances(sim: Simulation, options: PerkOptions, confidence: PerksConfidence) -> PerkOptions:
    sperks = perk_category_option_chances(sim, options, confidence, STANDARD_PERKS, STANDARD_PERK_CHANCE)
    uperks =  perk_category_option_chances(sim, options, confidence, ULTIMATE_PERKS, ULTIMATE_PERK_CHANCE)
    tperks = perk_category_option_chances(sim, options, confidence, TRADEOFF_PERKS, TRADEOFF_PERK_CHANCE)
    return (sperks + uperks + tperks).inorm()


def perk_option_set_chances(sim: Simulation, confidence: PerksConfidence) -> PerkOptions:
    options = PerkOptions()
    options_count = sim.perk_option_quantity()
    if confidence.count == 0:
        options.options[sim.first_perk_choice] = 1.0
        options_count -= 1
    for _ in range(0, options_count):
        options += perk_option_chances(sim, options, confidence)
    return options


def active_perks_confidence(sim: Simulation, confidence: PerksConfidence) -> PerksConfidence:
    options = perk_option_set_chances(sim, confidence)

    lower_priority = set(ALL_PERKS.keys())
    for high_perk in sim.perk_priority_order:
        lower_priority.remove(high_perk)
        for lower_perk in lower_priority:
            options.options[lower_perk] *= (1.0 - options.options[high_perk])
    options.inorm()

    next_confidence = copy.deepcopy(confidence)
    next_confidence.count += 1
    for perk, probseq in next_confidence.perks.items():
        for i in range(0, min(len(probseq), next_confidence.count)):
            next_confidence.perks[perk][i] += (1.0 - probseq[i]) * options.options[perk]
    return next_confidence


def active_perks_confidence_sequence(sim: Simulation) -> Iterator[PerksConfidence]:
    confidence = PerksConfidence()
    yield confidence
    for _ in range(0, sum(ALL_PERKS.values())):
        confidence = active_perks_confidence(sim, confidence)
        yield confidence


def spawn_rate_index(sim: Simulation, wave: int):
    spawn_rate_row = (
        SPAWN_RATE_WAVES
        if sim.wave_accelerator is None
        else WAVE_ACCELERATOR_MASTERY_TABLE[sim.wave_accelerator]
    )
    index = max(i for i, min_wave in enumerate(spawn_rate_row) if min_wave <= wave)
    assert 0 <= index < len(SPAWN_RATE_SEQUENCE), f"Invalid spawn rate index: {index}"
    return index


def elite_spawn_count(sim: Simulation, wave: int) -> float:
    def index(table: list[int], value: int) -> int:
        return bisect.bisect(table, value, lo=1) - 1

    single_index = index(ELITE_SINGLE_SPAWN_WAVES_TABLE[sim.tier - 1], wave)
    double_index = index(ELITE_DOUBLE_SPAWN_WAVES_TABLE[sim.tier - 1], wave)
    single_chance = ELITE_SPAWN_CHANCE_TABLE[single_index]
    double_chance = ELITE_SPAWN_CHANCE_TABLE[double_index]
    combined_chance = single_chance * (1.0 + double_chance)
    double_spawn = (
        0.0
        if sim.enemy_balance is None
        else ENEMY_BALANCE_MASTERY_TABLE[sim.enemy_balance]
    )
    return combined_chance * (1.0 + double_spawn)


def fleet_spawn_count(sim: Simulation, wave: int) -> float:
    min_wave = FLEET_MIN_WAVE_SPAWN_TABLE[sim.tier - 1]
    spawn_period = FLEET_SPAWN_PERIOD_WAVE_TABLE[sim.tier - 1]
    spawn_count = FLEET_SPAWN_COUNT_TABLE[sim.tier - 1]
    if min_wave is None or min_wave > wave:
        return 0
    assert spawn_period is not None
    assert spawn_count is not None

    if (wave - min_wave) % spawn_period == 0:
        return spawn_count
    return 0


def fleet_shard_reward(sim: Simulation, wave: int) -> float:
    index = bisect.bisect(FLEET_MODULE_SHARD_DROP_TABLE, wave, key=lambda x: x[0] if x is not None else 0)
    index = min(index, len(FLEET_MODULE_SHARD_DROP_TABLE) - 1)
    return FLEET_MODULE_SHARD_DROP_TABLE[index][1]


def simulate_wave(sim: Simulation, perks: Perks, wave: int) -> Events:
    spawn_index = spawn_rate_index(sim, wave)
    spawn_rate = SPAWN_RATE_SEQUENCE[spawn_index]
    common_spawns = WAVE_DURATION * spawn_rate * SPAWN_RATE_FACTOR
    elite_spawns = elite_spawn_count(sim, wave)
    fleet_spawns = fleet_spawn_count(sim, wave)
    fleet_children_count = (10 + 14) / 2 # Average of random number between 10-14

    # Boss spawns are binary, once every N waves based on tier.
    boss_period = TIER_BOSS_PERIOD[sim.tier - 1]
    boss_spawn = 1 if (wave % boss_period) == 0 else 0
    # Assuming "Package After Boss" lab, a package will guaranteed spawn every wave
    # following a boss spawn (ie, when `wave - 1` was a boss wave). Otherwise, the
    # chance is based on workshop, labs, and module effects.
    package_spawn = sim.package_chance
    if (wave - 1) % boss_period == 0:
        package_spawn = 1

    double_skip_chance = 0.0
    if sim.wave_skip is not None:
        # A double skip can only happen if a single skip is triggered.
        double_skip_chance = WAVE_SKIP_CHANCE * WAVE_SKIP_MASTERY_TABLE[sim.wave_skip]
    # Any given wave `w` can be skipped if:
    # 1. `w-2` triggered a double skip, or
    # 2. `w-2` did not trigger a double skip, but `w-1` triggered a single skip
    wave_skip_chance = double_skip_chance + WAVE_SKIP_CHANCE * (1 - double_skip_chance)

    free_upgrade_bonus = perks.free_upgrade_chance_bonus(sim)

    enemies = {
        **{
            name: common_spawns * row[spawn_index]
            for name, row in SPAWN_CHANCE_TABLE.items()
        },
        "boss": boss_spawn,
        "scatter": elite_spawns / 3,
        "vampire": elite_spawns / 3,
        "ray": elite_spawns / 3,
        "saboteur": fleet_spawns / 3,
        "commander": fleet_spawns / 3,
        "overcharge": fleet_spawns / 3,
    }
    enemies["fast"] += enemies["saboteur"] * fleet_children_count
    enemies["tank"] += enemies["commander"] * fleet_children_count
    enemies["ranged"] += enemies["overcharge"] * fleet_children_count

    return Events(
        wave=wave,
        wave_skip=wave_skip_chance,
        free_upgrades={
            name: chance + free_upgrade_bonus
            for name, chance in sim.free_upgrade_chances.items()
        },
        recovery_packages=package_spawn,
        enemy_level_skips=sim.enemy_level_skip_chances,
        enemies=enemies,
    )


def wave_skip_bonus_lerp(events: Events, noskip: float, skip: float) -> float:
    return (1 - events.wave_skip) * noskip + (events.wave_skip * skip)


def wave_skip_bonus_geom(events: Events, value: float) -> float:
    geom = sum(events.wave_skip ** i for i in range(1, 11))
    return value ** (1 + geom)


def calculate_coins(
    sim: Simulation, perks: Perks, events: Events, previous_events: Events, previous_rewards: Rewards
) -> float:
    coin_bonus = TIER_COIN_BONUS[sim.tier - 1] * perks.coin_bonus(sim)
    if sim.coin is not None:
        coin_bonus *= COIN_MASTERY_TABLE[sim.coin]
    if sim.bhd_bonus > 0:
        bhd_bonus = 1 + sim.bhd_bonus * sum(events.free_upgrades.values())
        coin_bonus *= wave_skip_bonus_geom(events, bhd_bonus)
    if sim.golden_combo > 0:
        # Calculate the number of enemies that died over the past two waves for the
        # golden combo exponent. Average the coin reward across the two waves.
        golden_enemies = events.total_enemy_count() + previous_events.total_enemy_count()
        coin_bonus *= ((1 + sim.golden_combo) ** golden_enemies) / 2

    orb_bonus = 1.0
    if sim.extra_orb is not None:
        orb_bonus *= 1 + ((EXTRA_ORB_MASTERY_TABLE[sim.extra_orb] - 1) * sim.orb_hits)

    coins_per_enemy = {
        name: drop * coin_bonus for name, drop in COIN_DROP_TABLE.items()
    }
    if sim.critical_coin is not None:
        coins_per_enemy["basic"] *= 1.0 + CRITICAL_COIN_MASTERY_TABLE[sim.critical_coin]

    coins = 0.0
    for name, count in events.enemies.items():
        coins += count * coins_per_enemy[name] * orb_bonus
    # The original scatter and each of its splits give the same amount of coins, but
    # only scatter splits struck by orbs give the EO# bonus. We'll assume that most
    # scatters make it inside the orb line before splitting, so only count the orb
    # bonus for the original scatter.
    coins += events.scatter_children_count() * coins_per_enemy["scatter"]
    return wave_skip_bonus_lerp(events, coins, previous_rewards.coins * WAVE_SKIP_BONUS)


def calculate_cells(
    sim: Simulation, events: Events, previous_rewards: Rewards
) -> float:
    # Elites drop a random number of cells between the min and max values for the tier.
    # Normally that's 1-TIER, but higher tiers have a floor above 1.
    cells_per_elite = (
        TIER_CELL_DROP_MIN[sim.tier - 1] + TIER_CELL_DROP_MAX[sim.tier - 1]
    ) / 2
    # Each elite *spawn* drops cells equally. Scatter splits all only count as a single
    # spawned elite. So even though we see 31 enemies killed per Scatter spawned, we
    # only get one "cell drop" event.
    total_elite_count = (
        events.enemies["scatter"] + events.enemies["vampire"] + events.enemies["ray"]
    )

    elite_cells = total_elite_count * cells_per_elite
    return wave_skip_bonus_lerp(
        events, elite_cells, previous_rewards.elite_cells * WAVE_SKIP_BONUS
    )


def calculate_rerolls(sim: Simulation, events: Events) -> float:
    rerolls_per_boss = TIER_REROLL_DROP[sim.tier - 1]
    boss_rerolls = events.enemies["boss"] * BOSS_REROLL_SHARD_DROP_CHANCE * rerolls_per_boss

    total_fleet_count = events.fleet_enemy_count()
    rerolls_per_fleet = FLEET_REROLL_SHARD_DROP_TABLE[sim.tier - 1]
    fleet_rerolls = total_fleet_count * FLEET_REROLL_SHARD_DROP_CHANCE * rerolls_per_fleet

    reroll_shards = boss_rerolls + fleet_rerolls

    if sim.cash is not None:
        # Each type of elite drops reroll shards with cash mastery.
        total_elite_count = events.elite_enemy_count()
        # Elites drop half as many reroll shards as bosses.
        rerolls_per_elite = rerolls_per_boss / 2
        elite_rerolls = (
            total_elite_count * CASH_MASTERY_TABLE[sim.cash] * rerolls_per_elite
        )
        # Elites do not drop reroll shards on skipped waves.
        reroll_shards += wave_skip_bonus_lerp(events, elite_rerolls, 0)

    return reroll_shards


def calculate_modules(sim: Simulation, events: Events) -> float:
    common_modules = events.enemies["boss"] * BOSS_COMMON_MODULE_DROP_CHANCE
    if sim.recovery_package is not None:
        # Recovery packages have a chance to provide modules.
        package_modules = (
            events.recovery_packages
            * RECOVERY_PACKAGE_CHANCE_MASTERY_TABLE[sim.recovery_package]
        )
        # Recovery packages do not provide modules on skipped waves.
        common_modules += wave_skip_bonus_lerp(events, package_modules, 0)
    module_shards = common_modules * COMMON_MODULE_VALUE

    rare_modules = events.enemies["boss"] * RARE_MODULE_DROP_CHANCE
    module_shards += rare_modules * RARE_MODULE_VALUE

    total_fleet_count = events.fleet_enemy_count()
    shards_per_fleet = fleet_shard_reward(sim, events.wave)
    fleet_shards = total_fleet_count * FLEET_MODULE_SHARD_DROP_CHANCE * shards_per_fleet
    module_shards += fleet_shards

    return module_shards


def calculate_rewards(
    sim: Simulation, perks: Perks, events: Events, previous_events: Events, previous_rewards: Rewards
) -> Rewards:
    return Rewards(
        coins=calculate_coins(sim, perks, events, previous_events, previous_rewards),
        elite_cells=calculate_cells(sim, events, previous_rewards),
        reroll_shards=calculate_rerolls(sim, events),
        module_shards=calculate_modules(sim, events),
    )


def simulate_run(sim: Simulation) -> Iterator[SimulationWaveResult]:
    elapsed_time = 0
    cumulative_events = Events()
    cumulative_rewards = Rewards()
    previous_events = Events()
    previous_rewards = Rewards()

    yield SimulationWaveResult(
        wave=0,
        elapsed_time=elapsed_time,
        cumulative_events=copy.deepcopy(cumulative_events),
        cumulative_rewards=copy.deepcopy(cumulative_rewards),
    )

    perk_estimator = PerkWaveEstimator(confidences=list(active_perks_confidence_sequence(sim)))
    perks = Perks()

    # Intro sprint

    intro_wave_count = max_intro_wave(sim)
    for wave in range(1, intro_wave_count):
        perks = perk_estimator.estimate(sim, wave, perks)
        events = simulate_wave(sim, perks, wave)
        # The only waves that don't skip during intro sprint are the 1st and every 10th.
        if wave == 1 or wave % 10 == 0:
            events.wave_skip = 0.0
            # Unskipped intro waves are guaranteed to have a boss.
            events.enemies["boss"] = 1
        else:
            events.wave_skip = 1.0
            # Skipped intro waves are guaranteed to not have a boss.
            events.enemies["boss"] = 0
        rewards = calculate_rewards(sim, perks, events, previous_events, previous_rewards)
        # No coins, rerolls, or modules are earned during intro sprint, and cells are
        # reduced to only 20%.
        rewards = Rewards(elite_cells=rewards.elite_cells * 0.2)
        wave_time = (WAVE_DURATION + WAVE_COOLDOWN) * (1 - events.wave_skip)
        wave_time /= (GAME_SPEED * perks.game_speed_factor(sim))

        previous_events = events
        cumulative_events += events
        previous_rewards = rewards
        cumulative_rewards += rewards
        elapsed_time += wave_time

        yield SimulationWaveResult(
            wave=wave,
            elapsed_time=elapsed_time,
            cumulative_events=copy.deepcopy(cumulative_events),
            cumulative_rewards=copy.deepcopy(cumulative_rewards),
        )

    # First regular wave after intro sprint (not skippable)

    perks = perk_estimator.estimate(sim, intro_wave_count, perks)
    events = simulate_wave(sim, perks, intro_wave_count)
    # The first wave after intro sprint is guaranteed not to skip.
    events.wave_skip = 0.0
    rewards = calculate_rewards(sim, perks, events, previous_events, previous_rewards)
    wave_time = WAVE_DURATION + WAVE_COOLDOWN
    wave_time /= (GAME_SPEED * perks.game_speed_factor(sim))

    previous_events = events
    cumulative_events += events
    previous_rewards = rewards
    cumulative_rewards += rewards
    elapsed_time += wave_time

    yield SimulationWaveResult(
        wave=intro_wave_count,
        elapsed_time=elapsed_time,
        cumulative_events=copy.deepcopy(cumulative_events),
        cumulative_rewards=copy.deepcopy(cumulative_rewards),
    )

    # Regular waves

    for wave in range(intro_wave_count + 1, sim.max_waves + 1):
        perks = perk_estimator.estimate(sim, wave, perks)
        events = simulate_wave(sim, perks, wave)
        rewards = calculate_rewards(sim, perks, events, previous_events, previous_rewards)
        wave_time = (WAVE_DURATION + WAVE_COOLDOWN) * (1 - events.wave_skip)
        wave_time /= (GAME_SPEED * perks.game_speed_factor(sim))

        previous_events = events
        cumulative_events += events
        previous_rewards = rewards
        cumulative_rewards += rewards
        elapsed_time += wave_time

        yield SimulationWaveResult(
            wave=wave,
            elapsed_time=elapsed_time,
            cumulative_events=copy.deepcopy(cumulative_events),
            cumulative_rewards=copy.deepcopy(cumulative_rewards),
        )


def evaluate_sims(
    sims: list[Simulation],
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    for sim in sims:
        if sim.skip:
            yield sim, None
            continue
        wave_results = list(simulate_run(sim))
        total = reward_value(sim, wave_results[-1].cumulative_rewards)
        yield sim, SimulationRunResult(wave_results=wave_results, total=total)


# Data normalization


def results_at_time(
    run_result: SimulationRunResult, elapsed_time: float
) -> tuple[SimulationWaveResult, SimulationWaveResult, float]:
    index = bisect.bisect_left(
        run_result.wave_results, elapsed_time, key=lambda x: x.elapsed_time
    )
    if index == 0:
        first_result = run_result.wave_results[0]
        return first_result, first_result, 0.0
    if index == len(run_result.wave_results):
        last_result = run_result.wave_results[-1]
        return last_result, last_result, 1.0
    lower_result = run_result.wave_results[index - 1]
    higher_result = run_result.wave_results[index]
    difference = elapsed_time - lower_result.elapsed_time
    basis = higher_result.elapsed_time - lower_result.elapsed_time
    assert basis != 0
    lambda_ = difference / basis
    return lower_result, higher_result, lambda_


def rewards_at_time(run_result: SimulationRunResult, elapsed_time: float) -> Rewards:
    lower_result, higher_result, lambda_ = results_at_time(run_result, elapsed_time)
    lower_rewards = lower_result.cumulative_rewards
    higher_rewards = higher_result.cumulative_rewards
    return lower_rewards + (higher_rewards - lower_rewards) * lambda_


def rewards_at_wave(run_result: SimulationRunResult, wave: int) -> Rewards:
    index = bisect.bisect_left(
        run_result.wave_results, wave, key=lambda x: x.wave
    )
    return run_result.wave_results[index].cumulative_rewards


def reward_value(sim: Simulation, rewards: Rewards) -> float:
    if sim.reward == "coins":
        return rewards.coins
    elif sim.reward == "cells":
        return rewards.elite_cells
    elif sim.reward == "rerolls":
        return rewards.reroll_shards
    elif sim.reward == "modules":
        return rewards.module_shards
    raise ValueError(f"Invalid reward: {sim.reward}")


def relative_rewards(lhs: Rewards, rhs: Rewards) -> Rewards:
    def relative_value(lhs: float, rhs: float) -> float:
        return 0.0 if rhs == 0 else (lhs / rhs - 1.0)

    return Rewards(
        coins=relative_value(lhs.coins, rhs.coins),
        elite_cells=relative_value(lhs.elite_cells, rhs.elite_cells),
        reroll_shards=relative_value(lhs.reroll_shards, rhs.reroll_shards),
        module_shards=relative_value(lhs.module_shards, rhs.module_shards),
    )


def truncate_sims_to_shortest(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    min_time = min(
        run_result.wave_results[-1].elapsed_time
        for _, run_result in sim_results
        if run_result is not None
    )
    for sim, run_result in sim_results:
        if run_result is None:
            yield sim, None
            continue
        index = bisect.bisect(
            run_result.wave_results, min_time, key=lambda x: x.elapsed_time
        )
        wave_results = run_result.wave_results[:index]
        total = reward_value(sim, wave_results[-1].cumulative_rewards)
        yield sim, dataclasses.replace(
            run_result, wave_results=wave_results, total=total
        )


def annotate_sims_vs_baseline(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
    baseline_sim_name: str,
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    baseline_sim, baseline_results = next(
        (sim, run_result)
        for sim, run_result in sim_results
        if sim.name == baseline_sim_name
    )
    assert baseline_results is not None
    baseline_value = reward_value(
        baseline_sim, baseline_results.wave_results[-1].cumulative_rewards
    )

    for sim, run_result in sim_results:
        if run_result is None:
            yield sim, None
            continue
        run_max = reward_value(sim, run_result.wave_results[-1].cumulative_rewards)
        relative = (run_max / baseline_value - 1.0) if baseline_value != 0 else 0.0
        yield sim, dataclasses.replace(run_result, relative=relative)


def annotate_sims_vs_stone_cost(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    for sim, run_result in sim_results:
        if run_result is None:
            yield sim, None
            continue
        stone_cost = sim.stone_cost()
        if run_result.relative is None or stone_cost == 0:
            yield sim, run_result
            continue

        value = run_result.relative / stone_cost
        yield sim, dataclasses.replace(run_result, roi=value)


def difference_sims_vs_baseline(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
    baseline_sim_name: str,
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    baseline_results = next(
        run_result for sim, run_result in sim_results
        if sim.name == baseline_sim_name
    )
    assert baseline_results is not None

    for sim, run_result in sim_results:
        if run_result is None:
            yield sim, None
            continue
        difference_results = []
        for wave_result in run_result.wave_results:
            baseline_rewards = rewards_at_time(
                baseline_results, wave_result.elapsed_time
            )
            differenced_rewards = wave_result.cumulative_rewards - baseline_rewards
            difference_results.append(
                dataclasses.replace(wave_result, cumulative_rewards=differenced_rewards)
            )
        yield sim, dataclasses.replace(run_result, wave_results=difference_results)


def normalize_sims_vs_elapsed(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]]
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    for sim, run_result in sim_results:
        if run_result is None:
            yield sim, None
            continue
        wave_results = iter(run_result.wave_results)
        normalized_results = [
            copy.deepcopy(next(wave_results)),
        ]
        for wave_result in wave_results:
            assert wave_result.elapsed_time != 0.0
            normalized_rewards = wave_result.cumulative_rewards * (
                1 / wave_result.elapsed_time
            )
            normalized_results.append(
                dataclasses.replace(wave_result, cumulative_rewards=normalized_rewards)
            )
        yield sim, dataclasses.replace(run_result, wave_results=normalized_results)


def normalize_sims_vs_baseline(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
    baseline_sim_name: str,
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    baseline_results = next(
        run_result for sim, run_result in sim_results if sim.name == baseline_sim_name
    )
    assert baseline_results is not None

    for sim, run_result in sim_results:
        if run_result is None:
            yield sim, None
            continue
        normalized_results = []
        relative = 0.0
        for wave_result in run_result.wave_results:
            baseline_rewards = rewards_at_time(
                baseline_results, wave_result.elapsed_time
            )
            normalized_rewards = relative_rewards(
                wave_result.cumulative_rewards, baseline_rewards
            )
            relative = reward_value(sim, normalized_rewards)
            normalized_results.append(
                dataclasses.replace(wave_result, cumulative_rewards=normalized_rewards)
            )
        yield sim, dataclasses.replace(
            run_result, wave_results=normalized_results, relative=relative
        )


def normalize_sims_vs_stone_cost(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    for sim, run_result in sim_results:
        if run_result is None:
            yield sim, None
            continue
        normalized_results = []
        stone_cost = sim.stone_cost()
        roi = None
        if stone_cost == 0:
            for wave_result in run_result.wave_results:
                normalized_results.append(
                    dataclasses.replace(wave_result, cumulative_rewards=Rewards())
                )
        else:
            for wave_result in run_result.wave_results:
                factor = 1 / stone_cost
                roi = reward_value(sim, wave_result.cumulative_rewards) * factor
                normalized_rewards = wave_result.cumulative_rewards * factor
                normalized_results.append(
                    dataclasses.replace(
                        wave_result, cumulative_rewards=normalized_rewards
                    )
                )
        yield sim, dataclasses.replace(
            run_result, wave_results=normalized_results, roi=roi
        )


def extend_sims(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
) -> Iterator[tuple[Simulation, SimulationRunResult | None]]:
    max_time = max(
        run_result.wave_results[-1].elapsed_time
        for _, run_result in sim_results
        if run_result is not None
    )
    for sim, run_result in sim_results:
        if run_result is None:
            yield sim, None
            continue
        wave_time = WAVE_DURATION + WAVE_COOLDOWN
        total_rewards = run_result.wave_results[-1].cumulative_rewards
        total_events = run_result.wave_results[-1].cumulative_events
        duration = run_result.wave_results[-1].elapsed_time
        avg_rewards = Rewards(
                    coins=total_rewards.coins / duration * wave_time,
                    elite_cells=total_rewards.elite_cells / duration * wave_time,
                    module_shards=total_rewards.module_shards / duration * wave_time,
                    reroll_shards=total_rewards.reroll_shards / duration * wave_time,
                )
        extended_waves = copy.deepcopy(run_result.wave_results)
        while extended_waves[-1].elapsed_time < max_time:
            last_wave = copy.deepcopy(extended_waves[-1])
            last_wave.cumulative_rewards.coins += avg_rewards.coins
            last_wave.cumulative_rewards.elite_cells += avg_rewards.elite_cells
            last_wave.cumulative_rewards.module_shards += avg_rewards.module_shards
            last_wave.cumulative_rewards.reroll_shards += avg_rewards.reroll_shards
            last_wave.elapsed_time += wave_time

            extended_waves.append(last_wave)

        total = reward_value(sim, extended_waves[-1].cumulative_rewards)
        yield sim, dataclasses.replace(
            run_result, wave_results=extended_waves, total=total
        )


def normalize_sims(
    args: argparse.Namespace,
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
    baseline_sim_name: str,
) -> list[tuple[Simulation, SimulationRunResult | None]]:
    if args.truncate:
        sim_results = list(truncate_sims_to_shortest(sim_results))
    elif args.extend:
        sim_results = list(extend_sims(sim_results))
    if args.elapsed:
        sim_results = list(normalize_sims_vs_elapsed(sim_results))
    if args.relative:
        sim_results = list(normalize_sims_vs_baseline(sim_results, baseline_sim_name))
        if args.roi:
            sim_results = list(normalize_sims_vs_stone_cost(sim_results))
        else:
            sim_results = list(annotate_sims_vs_stone_cost(sim_results))
    else:
        sim_results = list(annotate_sims_vs_baseline(sim_results, baseline_sim_name))
        sim_results = list(annotate_sims_vs_stone_cost(sim_results))
    if args.difference:
        sim_results = list(difference_sims_vs_baseline(sim_results, baseline_sim_name))

    return sim_results


# Simulation config factories


def make_sim(args: argparse.Namespace) -> Simulation:
    return Simulation(
        tier=args.tier,
        orb_hits=args.orb_hits,
        reward=args.reward,
        sum_total_stone_cost=args.sum_total_stone_cost,
        bhd_bonus=(args.bhd / 100),
        golden_combo=(args.golden_combo / 100),
        free_upgrade_chances={
            "attack": (args.freeup_chance[0] / 100),
            "defense": (args.freeup_chance[1] / 100),
            "utility": (args.freeup_chance[2] / 100),
        },
        cash=args.cash,
        coin=args.coin,
        critical_coin=args.critical_coin,
        enemy_balance=args.enemy_balance,
        extra_orb=args.extra_orb,
        intro_sprint=args.intro_sprint,
        recovery_package=args.recovery_package,
        wave_accelerator=args.wave_accelerator,
        wave_skip=args.wave_skip,
    )


def tiers_sim(sim: Simulation, tier: int, max_wave: int) -> Simulation:
    return dataclasses.replace(
        sim,
        name=f"T{tier}:W{max_wave}",
        tier=tier,
        max_waves=max_wave,
    )


def waves_sim(sim: Simulation, max_wave: int) -> Simulation:
    return dataclasses.replace(
        sim,
        name=f"{max_wave} waves",
        max_waves=max_wave,
    )


def mastery_sim(
    sim: Simulation,
    mastery: str,
    level: int | None,
    rerolls_with_cash: int | None,
    omit_masteries: set[str] = set(),
) -> Simulation:
    sim = dataclasses.replace(sim, mastery=mastery, level=level, skip=(mastery in omit_masteries))

    if level is None:
        sim = dataclasses.replace(sim, name=f"{mastery}: locked")
    else:
        sim = dataclasses.replace(sim, name=f"{mastery}: level {level}")

    if mastery == "cash":
        return dataclasses.replace(sim, cash=level)
    elif mastery == "coin":
        return dataclasses.replace(sim, coin=level)
    elif mastery == "critical-coin":
        return dataclasses.replace(sim, critical_coin=level)
    elif mastery == "enemy-balance":
        return dataclasses.replace(sim, enemy_balance=level, cash=rerolls_with_cash)
    elif mastery == "extra-orb":
        return dataclasses.replace(sim, extra_orb=level)
    elif mastery == "intro-sprint":
        return dataclasses.replace(sim, intro_sprint=level, cash=rerolls_with_cash)
    elif mastery == "recovery-package":
        return dataclasses.replace(sim, recovery_package=level)
    elif mastery == "wave-accelerator":
        return dataclasses.replace(sim, wave_accelerator=level)
    elif mastery == "wave-skip":
        return dataclasses.replace(sim, wave_skip=level, cash=rerolls_with_cash)
    raise ValueError(f"Invalid mastery: {mastery}")


# Plotting


def calculate_margins(
    sim_results: list[tuple[Simulation, SimulationRunResult | None]]
) -> tuple[float, float]:
    # Produce bottom/top margins for a plot attempting to fit all the important data
    # points, which must include at least the last 2/3 of data points (by time).
    # If the global maximum is outside the last 2/3 of the data, the margins are dilated
    # to the lesser-value of the global maximum and 3-sigma over the mean. Sigma
    # calculation only considers data in the last 2/3 of the data set.
    # Margins are then dilated by 5% of the bottom-to-top range. If the global maximum
    # resides between the pre-dilated and dilated margin, the margin is dilated again to
    # +5% over the global maximum.

    assert len(sim_results) > 0

    # Find the 1/3 - 2/3 partition value of the data set.
    min_time = float("inf")
    max_time = float("-inf")
    for _, run_result in sim_results:
        if run_result is None:
            continue
        for wave_result in run_result.wave_results:
            min_time = min(min_time, wave_result.elapsed_time)
            max_time = max(max_time, wave_result.elapsed_time)
    time_partition = (max_time - min_time) / 3

    # Find the minmax of the data set globally and within the last 2/3 time range.
    min_value = float("inf")
    max_value = float("-inf")
    min_inlier = float("inf")
    max_inlier = float("-inf")
    values = []
    for sim, run_result in sim_results:
        if run_result is None:
            continue
        for wave_result in run_result.wave_results:
            value = reward_value(sim, wave_result.cumulative_rewards)
            min_value = min(min_value, value)
            max_value = max(max_value, value)
            if wave_result.elapsed_time >= time_partition:
                min_inlier = min(min_inlier, value)
                max_inlier = max(max_inlier, value)
                values.append(value)

    # Calculate mean and sigma over the last 2/3 of the data set.
    mean = np.mean(values)
    stddev = np.std(values)
    sigma = mean - 3.0 * stddev

    # Dilate the inliers up to 3-sigma over the mean to reach the global min/max.
    bottom = min(min_inlier, max(min_value, mean - sigma))
    top = max(max_inlier, min(max_value, mean + sigma))
    margin = (top - bottom) * 0.05

    # Dilate the data range by up to 5% to reach the global min/max.
    if min_value > bottom - margin:
        bottom = min_value
    if max_value < top + margin:
        top = max_value
    margin = min(0.05, (top - bottom) * 0.05)

    # Dilate the margins by 5% of the included data range.
    return bottom - margin, top + margin


def interesting_waves(sim: Simulation) -> set[int]:
    return {
        # Regular gameplay
        *(0, 1, *range(10, 101, 10), sim.max_waves),
        *SPAWN_RATE_WAVES,
        *range(100, sim.max_waves + 1, 100),
        # Enemy balance mastery
        *{wave - 1 for row in ELITE_SINGLE_SPAWN_WAVES_TABLE for wave in row},
        *{wave - 1 for row in ELITE_DOUBLE_SPAWN_WAVES_TABLE for wave in row},
        *{wave for row in ELITE_SINGLE_SPAWN_WAVES_TABLE for wave in row},
        *{wave for row in ELITE_DOUBLE_SPAWN_WAVES_TABLE for wave in row},
        # Intro sprint mastery
        *{wave - 1 for wave in range(10, max(INTRO_SPRINT_MASTERY_TABLE) + 1, 10)},
        *{wave for wave in range(10, max(INTRO_SPRINT_MASTERY_TABLE) + 1, 10)},
        *{wave + 1 for wave in [100, *INTRO_SPRINT_MASTERY_TABLE]},
        # Wave accelerator mastery
        *{wave - 1 for row in WAVE_ACCELERATOR_MASTERY_TABLE for wave in row},
        *{wave for row in WAVE_ACCELERATOR_MASTERY_TABLE for wave in row},
    }


def plot_sim_results(
    args: argparse.Namespace,
    title: str,
    sim_results: list[tuple[Simulation, SimulationRunResult | None]],
) -> Plot:
    ylabel = args.reward
    if args.elapsed:
        ylabel += f" per hour"
    if args.difference:
        ylabel += f" difference"
    if args.relative:
        ylabel += f" relative"
    if args.roi:
        ylabel += f" per stone"

    plot = Plot(title=title, xlabel="Elapsed time (h)", ylabel=ylabel)
    for sim, run_result in sim_results:
        if run_result is None:
            plot.lines.append(None)
            continue

        line = PlotLine(
            name=sim.name,
            mastery=sim.mastery,
            relative=run_result.relative,
            roi=run_result.roi,
        )
        waves_to_plot = interesting_waves(sim)
        for wave_result in run_result.wave_results:
            if wave_result.wave not in waves_to_plot:
                continue
            value = reward_value(sim, wave_result.cumulative_rewards)
            line.xs.append(wave_result.elapsed_time / 3600)
            line.ys.append(value)
        plot.lines.append(line)

    if args.relative and args.crop:
        plot.bottom, plot.top = calculate_margins(sim_results)

    return plot


def render_plot(plot: Plot, /, show: bool = True, output: str | None = None):
    _, ax = plt.subplots(1, 1, figsize=(12, 10))

    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, line in enumerate(plot.lines):
        if line is None:
            continue
        label = f"{line.name}"
        if line.relative is not None:
            label += f"\n({line.relative:+.2%})"
        if line.roi is not None:
            label += f"\n[{line.roi:.6%}/stone]"
        ax.plot(
            line.xs,
            line.ys,
            label=label,
            color=colors[i % len(colors)],
            linewidth=2,
            marker="o",
            markersize=4,
        )

    if plot.top is not None:
        ax.set_ylim(top=plot.top)
    if plot.bottom is not None:
        ax.set_ylim(bottom=plot.bottom)

    ax.set_xlabel(plot.xlabel)
    ax.set_ylabel(plot.ylabel)
    ax.set_title(plot.title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {output}")

    if show:
        plt.show()


def si_format(value: float) -> str:
    formatter = sciform.Formatter(
        exp_mode="engineering",
        exp_format="prefix",
        round_mode="dec_place",
        ndigits=2,
    )
    return formatter(value)


def print_sim_results(sim_results: list[tuple[Simulation, SimulationRunResult | None]]):
    for sim, run_result in sim_results:
        if run_result is None:
            print(f"{sim.name}: skipped")
            continue
        assert run_result.total is not None
        message = f"{sim.name}: total={si_format(run_result.total)}"
        if run_result.relative is not None:
            message += f", relative={run_result.relative:.3%}"
            if run_result.roi is not None:
                message += f", roi={run_result.roi:.6%}"
        print(message)


# Subcommand implementations


def subcommand_tiers(args: argparse.Namespace) -> Plot:
    args.mastery = None
    args.tier = None
    convert_mastery_args(args)
    config = make_sim(args)

    sims = [tiers_sim(config, tier, wave) for tier, wave in args.tiers]
    baseline_sim_name = sims[0].name
    args.tiers.sort()
    sims.sort(key=lambda sim: sim.name)
    sim_results = normalize_sims(args, list(evaluate_sims(sims)), baseline_sim_name)
    if args.print:
        print_sim_results(sim_results)

    title = ", ".join(
        [
            f"Simulating tiers {', '.join(f'T{tier}:W{wave}' for tier, wave in args.tiers)}",
            *common_args_description(args, baseline_sim_name),
        ]
    )
    return plot_sim_results(args, title, sim_results)


def subcommand_compare(args: argparse.Namespace) -> Plot:
    args.level = mastery_level(args.level)
    convert_mastery_args(args)
    config = make_sim(args)
    config.max_waves = args.wave

    baseline_sim_name = "baseline"
    baseline_sim = dataclasses.replace(config, name=baseline_sim_name)

    sims = [baseline_sim] + [
        mastery_sim(config, mastery, args.level, args.rerolls_with_cash, args.omit)
        for mastery in MASTERY_DISPLAY_NAMES.keys()
    ]
    sim_results = normalize_sims(args, list(evaluate_sims(sims)), baseline_sim_name)
    if args.print:
        print_sim_results(sim_results)

    title = ", ".join(
        [
            f"Comparing masteries at level {args.level}",
            *common_args_description(args, baseline_sim_name),
            f"for T{args.tier}W{args.wave}",
        ]
    )
    return plot_sim_results(args, title, sim_results)


def subcommand_mastery(args: argparse.Namespace) -> Plot:
    convert_mastery_args(args)
    config = make_sim(args)
    config.max_waves = args.wave

    sims = [
        mastery_sim(config, args.mastery, level, args.rerolls_with_cash)
        for level in MASTERY_LEVELS
    ]
    baseline_sim = sims[0]
    sim_results = normalize_sims(args, list(evaluate_sims(sims)), baseline_sim.name)
    if args.print:
        print_sim_results(sim_results)

    relative_to = (
        "locked" if baseline_sim.level is None else f"level {baseline_sim.level}"
    )
    title = ", ".join(
        [
            f"Comparing {MASTERY_DISPLAY_NAMES[args.mastery]}# levels",
            *common_args_description(args, relative_to),
            f"for T{args.tier}W{args.wave}",
        ]
    )
    return plot_sim_results(args, title, sim_results)


def subcommand_custom(args: argparse.Namespace) -> Plot:
    args.mastery = None
    args.tier = None
    convert_mastery_args(args)
    config = make_sim(args)

    sims = []
    raise NotImplementedError("Custom simulation is not implemented")

    baseline_sim_name = sims[0].name
    sim_results = normalize_sims(args, list(evaluate_sims(sims)), baseline_sim_name)
    if args.print:
        print_sim_results(sim_results)

    title = ", ".join(
        [
            f"TODO custom title",
            *common_args_description(args, baseline_sim_name),
        ]
    )
    return plot_sim_results(args, title, sim_results)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    # Simulate a sequence of tier/wave pairs with fixed mastery levels
    tiers_subparser = subparsers.add_parser("tiers")
    tiers_subparser.add_argument(
        "tiers",
        type=tier_and_wave_arg,
        nargs="+",
        help="Sequence of tier/wave pairs to simulate",
    )
    add_common_args(tiers_subparser)

    # Compare all masteries at a single level
    compare_subparser = subparsers.add_parser("compare")
    compare_subparser.add_argument("wave", type=int, help="Wave number to simulate")
    compare_subparser.add_argument(
        "--level",
        "-l",
        choices=MASTERY_LEVEL_NAMES,
        default="1",
        help="Compare all masteries at this level",
    )
    compare_subparser.add_argument(
        "--omit",
        choices=MASTERY_DISPLAY_NAMES.keys(),
        default=set(),
        help="Omit masteries from the comparison",
    )
    add_common_args(compare_subparser)

    # Compare all levels of a single mastery
    mastery_subparser = subparsers.add_parser("mastery")
    mastery_subparser.add_argument("wave", type=int, help="Wave number to simulate")
    mastery_subparser.add_argument(
        "mastery",
        choices=MASTERY_DISPLAY_NAMES.keys(),
        help="Compare all mastery levels of this mastery",
    )
    add_common_args(mastery_subparser)

    # Custom simulation
    custom_subparser = subparsers.add_parser("custom")
    add_common_args(custom_subparser)

    args = parser.parse_args()

    if not 0.0 <= args.orb_hits <= 1.0:
        parser.error("--orb-hits must be between 0.0 and 1.0")
    if args.relative and args.difference:
        parser.error("--relative and --difference are mutually exclusive")
    if args.truncate and args.extend:
        parser.error("--truncate and --extend are mutually exclusive")
    if not args.relative and args.roi:
        parser.error("--roi can only be used with --relative")

    if args.subcommand == "tiers":
        plot = subcommand_tiers(args)
    elif args.subcommand == "compare":
        plot = subcommand_compare(args)
    elif args.subcommand == "mastery":
        plot = subcommand_mastery(args)
    elif args.subcommand == "custom":
        plot = subcommand_custom(args)
    else:
        parser.error(f"Invalid subcommand: {args.subcommand}")
    render_plot(plot, show=args.plot, output=args.output)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
