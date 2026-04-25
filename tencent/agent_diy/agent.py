#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import math
import numpy as np
from kaiwudrl.interface.agent import BaseAgent
from agent_diy.model.model import Model
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.algorithm.algorithm import Algorithm

MAP_SIZE, MAX_SPEED, MAX_CD, MAX_BUFF = 128.0, 5.0, 2000.0, 50.0

def _norm(v, v_max, v_min=0.0):
    return float(np.clip((v - v_min) / (v_max - v_min + 1e-6), 0.0, 1.0))

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.START_LR)
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.last_min_dist = 0.5
        self.last_treasures = 0
        self.last_buffs = 0
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, list_obs_data=None, *args, **kwargs):
        self.last_min_dist = 0.5
        self.last_treasures = 0
        self.last_buffs = 0

    def observation_process(self, env_obs, extra_info=None):
        obs = env_obs["observation"]
        h_x, h_z = obs["frame_state"]["heroes"]["pos"]["x"], obs["frame_state"]["heroes"]["pos"]["z"]

        h_feat = np.array([_norm(h_x, MAP_SIZE), _norm(h_z, MAP_SIZE), 
                           _norm(obs["frame_state"]["heroes"]["flash_cooldown"], MAX_CD), 
                           _norm(obs["frame_state"]["heroes"]["buff_remaining_time"], MAX_BUFF)], dtype=np.float32)

        m_feats = []
        for i in range(2):
            monsters = obs["frame_state"].get("monsters", [])
            if i < len(monsters) and monsters[i].get("is_in_view", 0):
                m = monsters[i]
                dist = math.sqrt((h_x - m["pos"]["x"])**2 + (h_z - m["pos"]["z"])**2)
                m_feats.append(np.array([1.0, _norm(m["pos"]["x"], MAP_SIZE), _norm(m["pos"]["z"], MAP_SIZE), 
                                         _norm(m.get("speed", 1), MAX_SPEED), _norm(dist, MAP_SIZE*1.41)], dtype=np.float32))
            else:
                m_feats.append(np.zeros(5, dtype=np.float32))

        organs = obs["frame_state"].get("organs", [])
        def get_closest(sub_type, max_num):
            items = [o for o in organs if o.get("sub_type") == sub_type and o.get("status") == 1]
            for it in items: it['dist'] = math.sqrt((h_x - it["pos"]["x"])**2 + (h_z - it["pos"]["z"])**2)
            items.sort(key=lambda x: x['dist'])
            feats = []
            for i in range(max_num):
                if i < len(items):
                    it = items[i]
                    feats.append(np.array([1.0, _norm(it["pos"]["x"], MAP_SIZE), _norm(it["pos"]["z"], MAP_SIZE),
                                           _norm(it['dist'], MAP_SIZE*1.41), _norm(it.get("hero_relative_direction", 0), 8.0)], dtype=np.float32))
                else: feats.append(np.zeros(5, dtype=np.float32))
            return np.concatenate(feats)
        
        t_feat, b_feat = get_closest(1, 5), get_closest(2, 2)

        map_feat = np.zeros(81, dtype=np.float32)
        if obs["map_info"] is not None and len(obs["map_info"]) >= 21:
            center = len(obs["map_info"]) // 2
            idx = 0
            for r in range(center-4, center+5):
                for c in range(center-4, center+5):
                    if 0 <= r < len(obs["map_info"]) and 0 <= c < len(obs["map_info"][0]):
                        map_feat[idx] = float(obs["map_info"][r][c] != 0)
                    idx += 1

        legal_act = [1] * 16
        raw_act = obs["legal_action"]
        if isinstance(raw_act, list) and raw_act:
            if isinstance(raw_act[0], bool): legal_act = [int(raw_act[j]) for j in range(min(16, len(raw_act)))]
            else: legal_act = [1 if j in {int(a) for a in raw_act if int(a) < 16} else 0 for j in range(16)]
        if sum(legal_act) == 0: legal_act = [1] * 16

        step_n = _norm(obs["step_no"], obs["env_info"].get("max_step", 1000))
        feature = np.concatenate([h_feat, m_feats[0], m_feats[1], t_feat, b_feat, map_feat, np.array(legal_act, dtype=np.float32), np.array([step_n, step_n], dtype=np.float32)])

        cur_min = min([m[4] for m in m_feats if m[0] > 0] + [1.0])
        dist_reward = 0.2 * (cur_min - self.last_min_dist)
        self.last_min_dist = cur_min

        cur_t, cur_b = obs["env_info"].get("treasures_collected", 0), obs["env_info"].get("collected_buff", 0)
        t_reward = 10.0 * (cur_t - self.last_treasures) if cur_t > self.last_treasures else 0.0
        b_reward = 3.0 * (cur_b - self.last_buffs) if cur_b > self.last_buffs else 0.0
        self.last_treasures, self.last_buffs = cur_t, cur_b

        return ObsData(feature=list(feature), legal_action=legal_act), {"reward": 0.05 + dist_reward + t_reward + b_reward}

    def predict(self, list_obs_data):
        feat, legal = np.array(list_obs_data[0].feature, dtype=np.float32), np.array(list_obs_data[0].legal_action, dtype=np.float32)
        self.model.set_eval_mode()
        with torch.no_grad():
            logits, value = self.model(torch.tensor(np.array([feat])).to(self.device))
        
        logits_np, value_np = logits.cpu().numpy()[0], value.cpu().numpy()[0]
        safe_logits = logits_np - 1e10 * (1.0 - legal)
        exp_logits = np.exp(safe_logits - np.max(safe_logits)) * legal
        prob = exp_logits / (np.sum(exp_logits) + 1e-9)

        prob = prob.astype(np.float64)
        prob /= np.sum(prob)

        action = int(np.argmax(np.random.multinomial(1, prob)))
        return [ActData(action=[action], d_action=[int(np.argmax(prob))], prob=list(prob), value=value_np)]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        return self.action_process(self.predict([obs_data])[0], is_stochastic=False)

    def learn(self, list_sample_data): return self.algorithm.learn(list_sample_data)
    def action_process(self, act_data, is_stochastic=True): return int((act_data.action if is_stochastic else act_data.d_action)[0])
    def save_model(self, path=None, id="1"): torch.save({k: v.clone().cpu() for k, v in self.model.state_dict().items()}, f"{path}/model.ckpt-{id}.pkl")
    def load_model(self, path=None, id="1"):
        try: self.model.load_state_dict(torch.load(f"{path}/model.ckpt-{id}.pkl", map_location=self.device))
        except: pass