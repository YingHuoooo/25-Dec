#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import collections
import numpy as np

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
GRID = 21
CENTER = 10

def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

def _calc_flash_radar(map_info):
    radar = np.zeros(8, dtype=np.float32)
    if map_info is None or len(map_info) == 0:
        return radar
    offsets = [(0, 10), (-8, 8), (-10, 0), (-8, -8), (0, -10), (8, -8), (10, 0), (8, 8)]
    for i, (dr, dc) in enumerate(offsets):
        tr, tc = CENTER + dr, CENTER + dc
        if 0 <= tr < GRID and 0 <= tc < GRID:
            if map_info[tr][tc] != 0:
                radar[i] = 1.0
    return radar

def _world_to_grid(entity_x, entity_z, hero_x, hero_z):
    gr = CENTER + round(entity_z - hero_z)
    gc = CENTER + round(entity_x - hero_x)
    return int(gr), int(gc)

def _place_gaussian(channel, row, col, sigma=1.5):
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = row + dr, col + dc
            if 0 <= r < GRID and 0 <= c < GRID:
                val = np.exp(-(dr*dr + dc*dc) / (2 * sigma * sigma))
                channel[r, c] = max(channel[r, c], val)

def _calc_corridor(map_info):
    corridor = np.zeros(8, dtype=np.float32)
    if map_info is None or len(map_info) == 0:
        return corridor
    dirs = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    for i, (dr, dc) in enumerate(dirs):
        length = 0
        r, c = CENTER, CENTER
        for _ in range(10):
            r, c = r + dr, c + dc
            if 0 <= r < GRID and 0 <= c < GRID and map_info[r][c] != 0:
                length += 1
            else:
                break
        corridor[i] = length / 10.0
    return corridor

def _calc_openness(map_info):
    if map_info is None or len(map_info) == 0:
        return 0.0
    count = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = CENTER + dr, CENTER + dc
            if 0 <= r < GRID and 0 <= c < GRID and map_info[r][c] != 0:
                count += 1
    return count / 25.0

def _is_dead_end(map_info):
    if map_info is None or len(map_info) == 0:
        return 0.0
    blocked = 0
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        r, c = CENTER + dr, CENTER + dc
        if not (0 <= r < GRID and 0 <= c < GRID and map_info[r][c] != 0):
            blocked += 1
    return 1.0 if blocked >= 3 else 0.0

def _calc_encirclement(monsters, hero_x, hero_z):
    if len(monsters) < 2:
        return 0.0
    m0, m1 = monsters[0]["pos"], monsters[1]["pos"]
    v0x, v0z = m0["x"] - hero_x, m0["z"] - hero_z
    v1x, v1z = m1["x"] - hero_x, m1["z"] - hero_z
    len0 = np.sqrt(v0x*v0x + v0z*v0z) + 1e-6
    len1 = np.sqrt(v1x*v1x + v1z*v1z) + 1e-6
    cos_a = (v0x*v1x + v0z*v1z) / (len0 * len1)
    return float(np.clip((1.0 - cos_a) / 2.0, 0, 1))

def _calc_safe_escape(map_info, monsters, hero_x, hero_z):
    dirs = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    safe_count = 0
    safe_indices = []
    for i, (dr, dc) in enumerate(dirs):
        r, c = CENTER + dr, CENTER + dc
        if not (0 <= r < GRID and 0 <= c < GRID and map_info[r][c] != 0):
            continue
        away = True
        for m in monsters:
            mp = m["pos"]
            mx, mz = mp["x"] - hero_x, mp["z"] - hero_z
            if dc * mx + dr * mz > 0:
                away = False
                break
        if away:
            safe_count += 1
            safe_indices.append(i)
    return safe_count / 8.0, safe_indices


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist = 1.0
        self.last_min_treasure_dist = 1.0
        self.last_min_buff_dist = 1.0
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_pos = None
        self.recent_positions = collections.deque(maxlen=30)
        self.stuck_count = 0
        self.last_openness = 1.0
        self.cur_safe_escape_dirs = 8

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x, hero_z = hero_pos["x"], hero_pos["z"]
        flash_cd = hero.get("flash_cooldown", 0)
        monsters = frame_state.get("monsters", [])
        organs = frame_state.get("organs", [])

        # ========== CNN: 4-channel 21x21 map ==========
        map_channels = np.zeros((4, GRID, GRID), dtype=np.float32)

        # Ch0: passable terrain (map_info != 0 = passable)
        if map_info and len(map_info) >= GRID:
            arr = np.array(map_info, dtype=np.float32)
            map_channels[0] = (arr != 0).astype(np.float32)

        # Ch1: monster heatmap
        for m in monsters:
            mp = m["pos"]
            gr, gc = _world_to_grid(mp["x"], mp["z"], hero_x, hero_z)
            if 0 <= gr < GRID and 0 <= gc < GRID:
                _place_gaussian(map_channels[1], gr, gc, sigma=2.0)

        # Ch2: treasure positions
        treasure_list = [o for o in organs if o["sub_type"] == 1 and o["status"] == 1]
        for t in treasure_list:
            tp = t["pos"]
            gr, gc = _world_to_grid(tp["x"], tp["z"], hero_x, hero_z)
            if 0 <= gr < GRID and 0 <= gc < GRID:
                map_channels[2, gr, gc] = 1.0

        # Ch3: buff positions
        buff_list = [o for o in organs if o["sub_type"] == 2 and o["status"] == 1]
        for b in buff_list:
            bp = b["pos"]
            gr, gc = _world_to_grid(bp["x"], bp["z"], hero_x, hero_z)
            if 0 <= gr < GRID and 0 <= gc < GRID:
                map_channels[3, gr, gc] = 1.0

        cnn_flat = map_channels.flatten()  # 1764

        # ========== MLP: scalar features (55 dim) ==========
        # hero (4)
        hero_feat = np.array([
            _norm(hero_x, MAP_SIZE), _norm(hero_z, MAP_SIZE),
            _norm(flash_cd, MAX_FLASH_CD),
            _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION),
        ], dtype=np.float32)

        # monsters (5+5=10)
        monster_feats = []
        cur_min_monster_dist = 1.0
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]; mp = m["pos"]
                dx = (mp["x"] - hero_x) / MAP_SIZE
                dz = (mp["z"] - hero_z) / MAP_SIZE
                dist = np.sqrt(dx*dx + dz*dz) + 1e-6
                cur_min_monster_dist = min(cur_min_monster_dist, dist)
                monster_feats.append(np.array([
                    1.0, dx/dist, dz/dist,
                    _norm(m.get("speed", 1), MAX_MONSTER_SPEED),
                    np.clip(dist, 0, 1.5)
                ], dtype=np.float32))
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # treasures nearest 2 (6)
        cur_min_treasure_dist = 1.0
        treasure_feat = np.zeros(6, dtype=np.float32)
        if treasure_list:
            dists = []
            for t in treasure_list:
                tp = t["pos"]
                dx = (tp["x"] - hero_x) / MAP_SIZE
                dz = (tp["z"] - hero_z) / MAP_SIZE
                dists.append((np.sqrt(dx*dx+dz*dz), dx, dz))
            dists.sort()
            for i in range(min(2, len(dists))):
                d, dx, dz = dists[i]
                if i == 0:
                    cur_min_treasure_dist = np.clip(d, 0, 1.5)
                treasure_feat[i*3:(i+1)*3] = [dx/(d+1e-6), dz/(d+1e-6), np.clip(d, 0, 1.5)]

        # buffs nearest 2 (6)
        cur_min_buff_dist = 1.0
        buff_feat = np.zeros(6, dtype=np.float32)
        if buff_list:
            dists = []
            for b in buff_list:
                bp = b["pos"]
                dx = (bp["x"] - hero_x) / MAP_SIZE
                dz = (bp["z"] - hero_z) / MAP_SIZE
                dists.append((np.sqrt(dx*dx+dz*dz), dx, dz))
            dists.sort()
            for i in range(min(2, len(dists))):
                d, dx, dz = dists[i]
                if i == 0:
                    cur_min_buff_dist = np.clip(d, 0, 1.5)
                buff_feat[i*3:(i+1)*3] = [dx/(d+1e-6), dz/(d+1e-6), np.clip(d, 0, 1.5)]

        # progress (2)
        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, 1.0 - step_norm], dtype=np.float32)

        # flash radar (8)
        flash_radar = _calc_flash_radar(map_info)

        # danger (4)
        danger_feat = np.array([
            1.0 if cur_min_monster_dist < 0.2 else 0.0,
            1.0 if cur_min_monster_dist < 0.4 else 0.0,
            1.0 if self.step_no >= env_info.get("monster_speedup", 500) else 0.0,
            1.0 if len(monsters) >= 2 else 0.0,
        ], dtype=np.float32)

        # corridor (8)
        corridor = _calc_corridor(map_info)

        # openness (1) + dead_end (1) + stuck_count (1)
        cur_openness = _calc_openness(map_info)
        openness = np.array([cur_openness], dtype=np.float32)
        dead_end = np.array([_is_dead_end(map_info)], dtype=np.float32)
        stuck_feat = np.array([min(self.stuck_count / 10.0, 1.0)], dtype=np.float32)

        # pressure features (4): encirclement, safe_escape, best_escape_quality, space_shrink
        encirclement = np.array([_calc_encirclement(monsters, hero_x, hero_z)], dtype=np.float32)
        if map_info and len(map_info) >= GRID and len(monsters) > 0:
            safe_escape_ratio, safe_indices = _calc_safe_escape(map_info, monsters, hero_x, hero_z)
        else:
            safe_escape_ratio, safe_indices = 1.0, list(range(8))
        safe_escape = np.array([safe_escape_ratio], dtype=np.float32)
        best_eq = max((corridor[i] for i in safe_indices), default=0.0) if safe_indices else 0.0
        best_escape_quality = np.array([best_eq], dtype=np.float32)
        space_shrink = np.array([max(0.0, self.last_openness - cur_openness)], dtype=np.float32)
        self.last_openness = cur_openness

        # Concat scalar: 4+5+5+6+6+2+8+4+8+1+1+1+4 = 55
        vec_feat = np.concatenate([
            hero_feat, monster_feats[0], monster_feats[1],
            treasure_feat, buff_feat, progress_feat,
            flash_radar, danger_feat, corridor,
            openness, dead_end, stuck_feat,
            encirclement, safe_escape, best_escape_quality, space_shrink,
        ])

        # ========== Legal action mask ==========
        legal_action = [1] * 16
        can_flash = (flash_cd <= 0) and (cur_min_monster_dist < 0.4)
        if not can_flash:
            for i in range(8, 16):
                legal_action[i] = 0
        else:
            for i in range(8):
                if flash_radar[i] < 0.5:
                    legal_action[i + 8] = 0

        # ========== Final feature: CNN flat + scalar ==========
        feature = np.concatenate([cnn_flat, vec_feat])  # 1764 + 55 = 1819

        self.cur_safe_escape_dirs = int(safe_escape_ratio * 8)
        reward = self._calc_reward(
            hero, env_info, cur_min_monster_dist,
            cur_min_treasure_dist, cur_min_buff_dist, last_action
        )
        return feature, legal_action, [reward]

    def _calc_reward(self, hero, env_info, cur_min_monster_dist, cur_min_treasure_dist, cur_min_buff_dist, last_action):
        r = 0.0
        hero_x, hero_z = hero["pos"]["x"], hero["pos"]["z"]

        # 1. stuck / wall penalty
        if self.last_pos and hero_x == self.last_pos[0] and hero_z == self.last_pos[1]:
            r -= 0.3
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_pos = (hero_x, hero_z)

        # 2. repeat position penalty
        pos_key = (int(hero_x), int(hero_z))
        repeat_count = self.recent_positions.count(pos_key)
        if repeat_count > 0:
            r -= 0.1 * min(repeat_count, 3)
        self.recent_positions.append(pos_key)

        # 3. monster distance shaping
        dist_diff = cur_min_monster_dist - self.last_min_monster_dist
        if cur_min_monster_dist < 0.15:
            r -= 0.5
        elif cur_min_monster_dist < 0.3:
            r += 3.0 * max(dist_diff, 0)
            r -= 0.2 * max(-dist_diff, 0)
        else:
            r += 1.0 * max(dist_diff, 0)

        # 4. treasure collected
        cur_tc = hero.get("treasure_collected_count", 0)
        if cur_tc > self.last_treasure_count:
            r += 10.0

        # 5. treasure approach (conditional on not approaching monster)
        if cur_min_monster_dist > 0.2:
            treasure_gain = self.last_min_treasure_dist - cur_min_treasure_dist
            if treasure_gain > 0:
                monster_gain = self.last_min_monster_dist - cur_min_monster_dist
                if monster_gain <= 0:
                    r += 2.0 * treasure_gain
                else:
                    r += 0.5 * treasure_gain

        # 6. buff collected
        cur_bc = env_info.get("collected_buff", 0)
        if cur_bc > self.last_buff_count:
            r += 8.0

        # 7. buff held
        if hero.get("buff_remaining_time", 0) > 0:
            r += 0.3

        # 8. flash: escape quality assessment
        if last_action >= 8:
            if dist_diff > 0.1 and self.cur_safe_escape_dirs >= 3:
                r += 3.0
            elif dist_diff > 0.1:
                r += 1.5
            elif dist_diff > 0:
                r -= 0.5
            else:
                r -= 1.5

        # 9. pre-speedup buffer
        speedup_step = env_info.get("monster_speedup", 500)
        if speedup_step - 100 < self.step_no < speedup_step + 50:
            r += 2.0 * max(dist_diff, 0)

        r *= 0.1

        self.last_min_monster_dist = cur_min_monster_dist
        self.last_min_treasure_dist = cur_min_treasure_dist
        self.last_min_buff_dist = cur_min_buff_dist
        self.last_treasure_count = cur_tc
        self.last_buff_count = cur_bc

        return float(np.clip(r, -0.5, 2.0))
