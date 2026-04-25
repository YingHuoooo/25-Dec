#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
from agent_diy.feature.definition import SampleData, sample_process
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if not usr_conf: return

    last_save_time, episode_cnt, last_report = time.time(), 0, 0

    while True:
        env_obs = env.reset(usr_conf)
        if handle_disaster_recovery(env_obs, logger): continue
        
        agent.reset(list_obs_data=[env_obs])
        agent.load_model(id="latest")

        obs_data, _ = agent.observation_process(env_obs)
        collector, step, total_reward, done = [], 0, 0.0, False
        episode_cnt += 1

        while not done:
            act_data = agent.predict([obs_data])[0]
            _, env_obs = env.step(agent.action_process(act_data))
            if handle_disaster_recovery(env_obs, logger): break

            done = env_obs["terminated"] or env_obs["truncated"]
            _obs_data, remain_info = agent.observation_process(env_obs)
            
            reward = np.array([remain_info.get("reward", 0.0)], dtype=np.float32).reshape(-1)
            total_reward += float(reward[0])
            final_reward = np.zeros(1, dtype=np.float32)
            env_info = env_obs["observation"]["env_info"]
            
            if done:
                final_reward[0] = -50.0 if env_obs["terminated"] else 50.0
                if logger: logger.info(f"[OVER] ep:{episode_cnt} step:{step} sim_score:{env_info.get('total_score',0)} total_rew:{total_reward:.2f}")

            collector.append(SampleData(
                obs=np.array(obs_data.feature, dtype=np.float32),
                legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                act=np.array([act_data.action[0]], dtype=np.float32),
                reward=reward, 
                done=np.array([float(done)], dtype=np.float32),
                reward_sum=np.zeros(1, dtype=np.float32),
                value=np.array([act_data.value], dtype=np.float32).reshape(-1),
                next_value=np.zeros(1, dtype=np.float32), 
                advantage=np.zeros(1, dtype=np.float32),
                prob=np.array(act_data.prob, dtype=np.float32)
            ))

            if done:
                collector[-1].reward += final_reward
                now = time.time()
                if now - last_report >= 60 and monitor:
                    monitor.put_data({os.getpid(): {"sim_score": env_info.get("total_score", 0), "treasures_collected": env_info.get("treasures_collected", 0)}})
                    last_report = now
                
                processed_collector = sample_process(collector)
                agent.send_sample_data(processed_collector)
                break
                
            obs_data = _obs_data
            step += 1

        if time.time() - last_save_time >= 1800:
            agent.save_model()
            last_save_time = time.time()