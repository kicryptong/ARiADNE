import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from datetime import datetime

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to RL autonomous exploration!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

# 성능 모니터링을 위한 폴더 생성
performance_path = f'performance/{FOLDER_NAME}'
if not os.path.exists(performance_path):
    os.makedirs(performance_path)

# 1000회마다 모델 저장을 위한 폴더 생성
milestone_model_path = f'milestone_models/{FOLDER_NAME}'
if not os.path.exists(milestone_model_path):
    os.makedirs(milestone_model_path)


def save_milestone_model(models_dict, episode, performance_metrics):
    """1000회마다 모델을 별도로 저장"""
    milestone_path = f"{milestone_model_path}/model_episode_{episode}"
    if not os.path.exists(milestone_path):
        os.makedirs(milestone_path)
    
    # 모델 저장
    torch.save(models_dict, f"{milestone_path}/checkpoint.pth")
    
    # 성능 메트릭도 함께 저장
    with open(f"{milestone_path}/performance_summary.json", 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    
    print(f"Milestone model saved at episode {episode}")


def plot_performance_metrics(performance_history, episode):
    """성능 메트릭 시각화"""
    if len(performance_history['episodes']) < 2:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Performance - Episode {episode}', fontsize=16)
    
    episodes = performance_history['episodes']
    
    # 1. Reward 그래프
    axes[0, 0].plot(episodes, performance_history['rewards'], 'b-', alpha=0.7)
    axes[0, 0].plot(episodes, performance_history['reward_ma'], 'r-', linewidth=2, label='Moving Average')
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Success Rate 그래프
    axes[0, 1].plot(episodes, performance_history['success_rates'], 'g-', alpha=0.7)
    axes[0, 1].plot(episodes, performance_history['success_rate_ma'], 'r-', linewidth=2, label='Moving Average')
    axes[0, 1].set_title('Success Rate')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Explored Rate 그래프
    axes[0, 2].plot(episodes, performance_history['explored_rates'], 'm-', alpha=0.7)
    axes[0, 2].plot(episodes, performance_history['explored_rate_ma'], 'r-', linewidth=2, label='Moving Average')
    axes[0, 2].set_title('Explored Rate')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Explored Rate')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Travel Distance 그래프
    axes[1, 0].plot(episodes, performance_history['travel_distances'], 'c-', alpha=0.7)
    axes[1, 0].plot(episodes, performance_history['travel_dist_ma'], 'r-', linewidth=2, label='Moving Average')
    axes[1, 0].set_title('Travel Distance')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Distance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Loss 그래프
    if len(performance_history['policy_losses']) > 0:
        axes[1, 1].plot(episodes[-len(performance_history['policy_losses']):], 
                       performance_history['policy_losses'], 'orange', alpha=0.7, label='Policy Loss')
        axes[1, 1].plot(episodes[-len(performance_history['q_losses']):], 
                       performance_history['q_losses'], 'purple', alpha=0.7, label='Q Loss')
        axes[1, 1].set_title('Training Losses')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    # 6. Entropy 그래프
    if len(performance_history['entropies']) > 0:
        axes[1, 2].plot(episodes[-len(performance_history['entropies']):], 
                       performance_history['entropies'], 'brown', alpha=0.7)
        axes[1, 2].set_title('Policy Entropy')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Entropy')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{performance_path}/performance_episode_{episode}.png', dpi=150, bbox_inches='tight')
    plt.close()


def calculate_moving_average(data, window=100):
    """이동평균 계산"""
    if len(data) < window:
        return [np.mean(data[:i+1]) for i in range(len(data))]
    else:
        ma = []
        for i in range(len(data)):
            if i < window:
                ma.append(np.mean(data[:i+1]))
            else:
                ma.append(np.mean(data[i-window+1:i+1]))
        return ma


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize neural networks
    global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    # 성능 모니터링을 위한 변수들
    performance_history = {
        'episodes': [],
        'rewards': [],
        'reward_ma': [],
        'success_rates': [],
        'success_rate_ma': [],
        'explored_rates': [],
        'explored_rate_ma': [],
        'travel_distances': [],
        'travel_dist_ma': [],
        'policy_losses': [],
        'q_losses': [],
        'entropies': []
    }

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha'] 
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs are available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(15):
        experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(8):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    node_inputs = torch.stack(rollouts[0]).to(device)
                    node_padding_mask = torch.stack(rollouts[1]).to(device)
                    edge_mask = torch.stack(rollouts[2]).to(device)
                    current_index = torch.stack(rollouts[3]).to(device)
                    current_edge = torch.stack(rollouts[4]).to(device)
                    edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    action = torch.stack(rollouts[6]).to(device)
                    reward = torch.stack(rollouts[7]).to(device)
                    done = torch.stack(rollouts[8]).to(device)
                    next_node_inputs = torch.stack(rollouts[9]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[10]).to(device)
                    next_edge_mask = torch.stack(rollouts[11]).to(device)
                    next_current_index = torch.stack(rollouts[12]).to(device)
                    next_current_edge = torch.stack(rollouts[13]).to(device)
                    next_edge_padding_mask = torch.stack(rollouts[14]).to(device)

                    observation = [node_inputs, node_padding_mask, edge_mask, current_index,
                                   current_edge, edge_padding_mask]
                    next_observation = [next_node_inputs, next_node_padding_mask, next_edge_mask,
                                        next_current_index, next_current_edge, next_edge_padding_mask]

                    # SAC
                    with torch.no_grad():
                        q_values1 = dp_q_net1(*observation)
                        q_values2 = dp_q_net2(*observation)
                        q_values = torch.min(q_values1, q_values2)

                    logp = dp_policy(*observation)
                    policy_loss = torch.sum(
                        (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                        dim=1).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = dp_policy(*next_observation)
                        next_q_values1 = dp_target_q_net1(*next_observation)
                        next_q_values2 = dp_target_q_net2(*next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q_batch = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()

                    q_values1 = dp_q_net1(*observation)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = dp_q_net2(*observation)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    target_q_update_counter += 1

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard and update performance history
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                
                # 성능 기록 업데이트
                avg_data = np.nanmean(training_data, axis=0)
                performance_history['episodes'].append(curr_episode)
                performance_history['rewards'].append(avg_data[0])
                performance_history['success_rates'].append(avg_data[-2])
                performance_history['explored_rates'].append(avg_data[-1])
                performance_history['travel_distances'].append(avg_data[-3])
                performance_history['policy_losses'].append(avg_data[2])
                performance_history['q_losses'].append(avg_data[3])
                performance_history['entropies'].append(avg_data[4])
                
                # 이동평균 계산
                performance_history['reward_ma'] = calculate_moving_average(performance_history['rewards'])
                performance_history['success_rate_ma'] = calculate_moving_average(performance_history['success_rates'])
                performance_history['explored_rate_ma'] = calculate_moving_average(performance_history['explored_rates'])
                performance_history['travel_dist_ma'] = calculate_moving_average(performance_history['travel_distances'])
                
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
            weights_set.append(policy_weights)

            # update the target q net
            if target_q_update_counter > 64:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # 1000회마다 마일스톤 모델 저장
            if curr_episode % 1000 == 0:
                checkpoint_data = {
                    "policy_model": global_policy_net.state_dict(),
                    "q_net1_model": global_q_net1.state_dict(),
                    "q_net2_model": global_q_net2.state_dict(),
                    "log_alpha": log_alpha,
                    "policy_optimizer": global_policy_optimizer.state_dict(),
                    "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                    "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                    "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                    "episode": curr_episode,
                }
                
                # 성능 요약 정보
                recent_performance = {}
                if len(performance_history['episodes']) > 0:
                    recent_performance = {
                        'latest_episode': curr_episode,
                        'avg_success_rate_last_100': np.mean(performance_history['success_rates'][-100:]) if len(performance_history['success_rates']) >= 100 else np.mean(performance_history['success_rates']),
                        'avg_explored_rate_last_100': np.mean(performance_history['explored_rates'][-100:]) if len(performance_history['explored_rates']) >= 100 else np.mean(performance_history['explored_rates']),
                        'avg_reward_last_100': np.mean(performance_history['rewards'][-100:]) if len(performance_history['rewards']) >= 100 else np.mean(performance_history['rewards']),
                        'timestamp': datetime.now().isoformat()
                    }
                
                save_milestone_model(checkpoint_data, curr_episode, recent_performance)

            # 성능 그래프 업데이트 (500회마다)
            if curr_episode % 500 == 0 and len(performance_history['episodes']) > 1:
                plot_performance_metrics(performance_history, curr_episode)
                
                # 성능 기록을 JSON 파일로 저장
                with open(f'{performance_path}/performance_history.json', 'w') as f:
                    json.dump(performance_history, f, indent=4)

            # save the model (기존 32회마다 저장)
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, travel_dist, success_rate, explored_rate = tensorboard_data

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)


if __name__ == "__main__":
    main()
