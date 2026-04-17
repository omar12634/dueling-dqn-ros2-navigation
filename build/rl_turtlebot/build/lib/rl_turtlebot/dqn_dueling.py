import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import signal
import sys
import time
import csv

# ═════════════════════════════════════════════════════════════════════════════
MAX_EPISODES   = 5000
MAX_STEPS      = 500
BATCH_SIZE     = 64
REPLAY_CAP     = 50_000
GAMMA          = 0.99
LR             = 1e-4
EPS_START      = 1.0
EPS_MIN        = 0.10
EPS_DECAY      = 0.995
TARGET_UPDATE  = 10
SAVE_EVERY     = 50

GOAL           = np.array([2.0, 2.0])
GOAL_R         = 0.55

DIST_FREE      = 1.00
DIST_ALERT     = 0.60
DIST_STOP      = 0.30

V_FAST         = 0.12
V_MEDIUM       = 0.07
V_SLOW         = 0.03
V_ANG          = 0.55

STUCK_STEPS    = 40
STUCK_DIST     = 0.01

STATE_DIM      = 5
ACTION_DIM     = 3

MODEL_PATH     = 'dueling_model.pth'
BACKUP_PATH    = 'dueling_model_backup.pth'
CSV_PATH       = 'dueling_results.csv'
PLOT_PATH      = 'dueling_training.png'


# ═════════════════════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    """
    Dueling DQN : Q(s,a) = V(s) + A(s,a) - mean(A)
    V(s)   = valeur de l'état
    A(s,a) = avantage de chaque action
    """
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, ACTION_DIM)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(ns), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# ═════════════════════════════════════════════════════════════════════════════
class DuelingDQNNode(Node):

    def __init__(self):
        super().__init__('dueling_dqn_nav')

        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.create_subscription(Odometry,  '/odom', self._on_odom, 10)
        self.reset_client = self.create_client(
            Empty, '/gazebo/reset_simulation'
        )

        self.scan          = None
        self.pos           = None
        self.yaw           = 0.0
        self.prev_dist     = None
        self.prev_pos      = None
        self._ready        = False
        self._resetting    = False
        self._stuck_count  = 0

        self.online = DuelingDQN()
        self.target = DuelingDQN()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=LR)
        self.buf = ReplayBuffer(REPLAY_CAP)

        self.eps     = EPS_START
        self.ep      = 0
        self.steps   = 0
        self.total_r = 0.0

        self.hist_r     = []
        self.hist_e     = []
        self.hist_s     = []
        self.hist_steps = []
        self.hist_rate  = []

        self._load_model()
        signal.signal(signal.SIGINT,  self._on_shutdown)
        signal.signal(signal.SIGTERM, self._on_shutdown)

        self.create_timer(0.10, self._loop)
        self.get_logger().info('=== Dueling DQN Nav — démarrage ===')

    def _on_scan(self, msg):
        r = np.array(msg.ranges, dtype=np.float32)
        self.scan = np.where(np.isfinite(r), r, float(msg.range_max))

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.pos = np.array([p.x, p.y], dtype=np.float32)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0*(q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.y*q.y + q.z*q.z)
        )

    def _reset_world(self):
        self._stop()
        self._resetting = True
        if self.reset_client.service_is_ready():
            self.reset_client.call_async(Empty.Request())
        time.sleep(1.5)
        self.scan         = None
        self.pos          = None
        self.prev_dist    = None
        self.prev_pos     = None
        self._ready       = False
        self._resetting   = False
        self._stuck_count = 0

    def _read_front(self):
        if self.scan is None:
            return 3.0, 3.0, 3.0
        n   = len(self.scan)
        i10 = int(n * 10 / 360)
        i30 = int(n * 30 / 360)
        front_idx = list(range(0, i10)) + list(range(n - i10, n))
        fl_idx    = list(range(i10, i30))
        fr_idx    = list(range(n - i30, n - i10))
        return (
            float(np.min(self.scan[fl_idx])),
            float(np.min(self.scan[front_idx])),
            float(np.min(self.scan[fr_idx])),
        )

    def _get_speed(self, front_dist):
        if front_dist > DIST_FREE:
            return V_FAST
        elif front_dist > DIST_ALERT:
            r = (front_dist - DIST_ALERT) / (DIST_FREE - DIST_ALERT)
            return V_MEDIUM + r * (V_FAST - V_MEDIUM)
        elif front_dist > DIST_STOP:
            r = (front_dist - DIST_STOP) / (DIST_ALERT - DIST_STOP)
            return V_SLOW + r * (V_MEDIUM - V_SLOW)
        return 0.0

    def _get_state(self):
        if self.scan is None or self.pos is None:
            return None
        fl, front, fr = self._read_front()
        dx, dy = GOAL - self.pos
        dist   = float(np.hypot(dx, dy))
        angle  = float(math.atan2(dy, dx) - self.yaw)
        angle  = (angle + math.pi) % (2*math.pi) - math.pi
        return np.array([
            min(fl,    3.0) / 3.0,
            min(front, 3.0) / 3.0,
            min(fr,    3.0) / 3.0,
            dist / 10.0,
            angle / math.pi,
        ], dtype=np.float32)

    def _act_and_publish(self, s) -> int:
        fl    = s[0] * 3.0
        front = s[1] * 3.0
        fr    = s[2] * 3.0
        angle = s[4] * math.pi
        v_lin = self._get_speed(front)

        if front < DIST_STOP:
            self.get_logger().warn(f'Trop proche ! front={front:.2f}m')
            return -1

        if front < DIST_ALERT:
            if fl >= fr:
                action, ang = 0, +V_ANG
            else:
                action, ang = 2, -V_ANG
            cmd = Twist()
            cmd.linear.x  = float(v_lin)
            cmd.angular.z = float(ang)
            self.cmd.publish(cmd)
            return action

        if fl < DIST_ALERT:
            cmd = Twist()
            cmd.linear.x  = float(v_lin)
            cmd.angular.z = -V_ANG
            self.cmd.publish(cmd)
            return 2

        if fr < DIST_ALERT:
            cmd = Twist()
            cmd.linear.x  = float(v_lin)
            cmd.angular.z = +V_ANG
            self.cmd.publish(cmd)
            return 0

        if random.random() >= self.eps:
            with torch.no_grad():
                t      = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                action = int(self.online(t).argmax().item())
        else:
            if abs(angle) < 0.20:
                action = 1
            elif angle > 0:
                action = random.choice([0, 1])
            else:
                action = random.choice([1, 2])

        ang_map = [+V_ANG, 0.0, -V_ANG]
        cmd = Twist()
        cmd.linear.x  = float(v_lin)
        cmd.angular.z = float(ang_map[action])
        self.cmd.publish(cmd)
        return action

    def _stop(self):
        for _ in range(5):
            self.cmd.publish(Twist())

    def _compute_reward(self, s):
        front = s[1] * 3.0
        dist  = s[3] * 10.0
        angle = s[4] * math.pi

        if front < DIST_STOP:
            return -200.0, True
        if dist < GOAL_R:
            return 500.0, True

        reward = 0.0
        if self.prev_dist is not None:
            delta = self.prev_dist - dist
            reward += delta * 150.0
            if delta < 0 and self.prev_dist < 1.5:
                reward -= 60.0
        self.prev_dist = dist

        if dist < 0.6:
            reward += 150.0
        elif dist < 1.0:
            reward += 60.0
        elif dist < 1.5:
            reward += 25.0
        elif dist < 2.0:
            reward += 8.0

        reward -= abs(angle) * 2.0

        if front < DIST_ALERT:
            reward -= (DIST_ALERT - front) / DIST_ALERT * 80.0

        if front > DIST_FREE:
            reward += 5.0

        return reward, False

    def _check_and_handle_stuck(self):
        if self.pos is None or self.prev_pos is None:
            return False
        moved = float(np.linalg.norm(self.pos - self.prev_pos))
        if moved < STUCK_DIST:
            self._stuck_count += 1
        else:
            self._stuck_count = max(0, self._stuck_count - 1)
        self.prev_pos = self.pos.copy()
        if self._stuck_count >= STUCK_STEPS:
            self.get_logger().warn(
                f'Immobile {self._stuck_count} steps → reset auto'
            )
            self._end_episode(False, 'BLOQUE')
            return True
        return False

    def _learn(self):
        if len(self.buf) < BATCH_SIZE * 2:
            return
        s, a, r, ns, d = self.buf.sample(BATCH_SIZE)
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a   = self.online(ns).argmax(1, keepdim=True)
            q_next   = self.target(ns).gather(1, best_a).squeeze(1)
            q_target = r + GAMMA * q_next * (1.0 - d)
        loss = nn.functional.mse_loss(q_pred, q_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()

    def _end_episode(self, success, reason=''):
        self._stuck_count = 0
        self.ep  += 1
        self.eps  = max(EPS_MIN, self.eps * EPS_DECAY)

        if self.ep % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.hist_r.append(self.total_r)
        self.hist_e.append(self.eps)
        self.hist_s.append(int(success))
        self.hist_steps.append(self.steps)

        w    = self.hist_s[-50:]
        rate = sum(w) / len(w)
        self.hist_rate.append(rate)

        self.get_logger().info(
            f'Ep {self.ep:4d}/{MAX_EPISODES} | '
            f'R={self.total_r:8.1f} | '
            f'Steps={self.steps:4d} | '
            f'eps={self.eps:.3f} | '
            f'Succes={rate:.0%} | '
            f'{"GOAL" if success else "ECHEC "+reason}'
        )

        if self.ep % SAVE_EVERY == 0:
            self._save_model(reason=f'ep{self.ep}')
            self._save_csv()
            self._save_plots()

        self.steps     = 0
        self.total_r   = 0.0
        self.prev_dist = None
        self.prev_pos  = None

        if success:
            self._stop()
            self.get_logger().info(
                '*** GOAL — arrêt 3s puis retour position initiale ***'
            )
            time.sleep(3.0)

        self._reset_world()

        if self.ep >= MAX_EPISODES:
            self._stop()
            self._save_model(reason='FIN')
            self._save_csv()
            self._save_plots()
            self.get_logger().info('=== DUELING DQN TERMINE ===')
            rclpy.shutdown()

    def _loop(self):
        if self._resetting:
            return

        s = self._get_state()
        if s is None:
            return

        if not self._ready:
            if self.pos is None:
                return
            dist_now = float(np.linalg.norm(GOAL - self.pos))
            _, front, _ = self._read_front()

            if dist_now < GOAL_R + 0.3:
                self.get_logger().warn(
                    f'Spawn proche goal ({dist_now:.2f}m) → reset'
                )
                self._reset_world()
                return

            self._ready    = True
            self.prev_dist = dist_now
            self.prev_pos  = self.pos.copy()
            self.get_logger().info(
                f'Ep {self.ep+1} démarré | '
                f'dist={dist_now:.2f}m | '
                f'front={front:.2f}m | '
                f'eps={self.eps:.3f}'
            )

        if self.pos is not None:
            dist_now = float(np.linalg.norm(GOAL - self.pos))
            if dist_now < GOAL_R and self.steps >= 3:
                self.get_logger().info(
                    f'*** GOAL ! dist={dist_now:.2f}m steps={self.steps} ***'
                )
                self._stop()
                self.total_r += 500.0
                self._end_episode(True, 'GOAL')
                return

        if self._check_and_handle_stuck():
            return

        action = self._act_and_publish(s)
        if action == -1:
            self._end_episode(False, 'TROP PROCHE')
            return

        reward, done = self._compute_reward(s)
        ns = self._get_state()
        if ns is None:
            return

        self.buf.push(s, action, reward, ns, float(done))
        self.total_r += reward
        self.steps   += 1
        self._learn()

        if done:
            ok = float(np.linalg.norm(GOAL - self.pos)) < GOAL_R
            self._end_episode(ok, 'GOAL' if ok else 'TROP PROCHE')
            return

        if self.steps >= MAX_STEPS:
            self._end_episode(False, 'TIMEOUT')

    def _save_plots(self):
        if not self.hist_r:
            return
        episodes = list(range(1, len(self.hist_r) + 1))
        w = 50
        r_smooth = [
            np.mean(self.hist_r[max(0, i-w):i+1])
            for i in range(len(self.hist_r))
        ]
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        fig.suptitle(
            'Dueling DQN — Courbes entraînement',
            fontsize=14, fontweight='bold'
        )

        axes[0].plot(episodes, self.hist_r,
                     color='lightblue', alpha=0.4,
                     linewidth=0.8, label='Reward brut')
        axes[0].plot(episodes, r_smooth,
                     color='blue', linewidth=2,
                     label=f'Moyenne {w} ep')
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        axes[0].set_title('Reward total par épisode')
        axes[0].set_xlabel('Épisode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, self.hist_e,
                     color='orange', linewidth=2)
        axes[1].fill_between(episodes, self.hist_e,
                             alpha=0.2, color='orange')
        axes[1].axhline(y=EPS_MIN, color='red', linestyle='--',
                        linewidth=1, label=f'EPS_MIN={EPS_MIN}')
        axes[1].set_title('Décroissance Epsilon')
        axes[1].set_xlabel('Épisode')
        axes[1].set_ylabel('Epsilon')
        axes[1].set_ylim(0, 1.1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(episodes, self.hist_rate,
                     color='green', linewidth=2,
                     label=f'Taux succès ({w} ep)')
        axes[2].fill_between(episodes, self.hist_rate,
                             alpha=0.2, color='green')
        axes[2].axhline(y=0.8, color='red', linestyle='--',
                        linewidth=1, label='Objectif 80%')
        axes[2].set_title('Taux de succès')
        axes[2].set_xlabel('Épisode')
        axes[2].set_ylabel('Taux')
        axes[2].set_ylim(0, 1.05)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        s_smooth = [
            np.mean(self.hist_steps[max(0, i-w):i+1])
            for i in range(len(self.hist_steps))
        ]
        axes[3].plot(episodes, self.hist_steps,
                     color='lightsalmon', alpha=0.4,
                     linewidth=0.8, label='Steps bruts')
        axes[3].plot(episodes, s_smooth,
                     color='red', linewidth=2,
                     label=f'Moyenne {w} ep')
        axes[3].set_title('Steps par épisode')
        axes[3].set_xlabel('Épisode')
        axes[3].set_ylabel('Steps')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f'Courbes → {PLOT_PATH}')

    def _save_csv(self):
        try:
            with open(CSV_PATH, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow([
                    'episode', 'reward', 'epsilon',
                    'success', 'steps', 'success_rate'
                ])
                for i in range(len(self.hist_r)):
                    wr.writerow([
                        i+1,
                        round(self.hist_r[i], 2),
                        round(self.hist_e[i], 4),
                        self.hist_s[i],
                        self.hist_steps[i],
                        round(self.hist_rate[i], 4),
                    ])
        except Exception as e:
            self.get_logger().error(f'CSV erreur: {e}')

    def _save_model(self, path=MODEL_PATH, reason='auto'):
        try:
            if os.path.exists(path) and path == MODEL_PATH:
                if os.path.exists(BACKUP_PATH):
                    os.remove(BACKUP_PATH)
                os.rename(MODEL_PATH, BACKUP_PATH)
            torch.save({
                'online_state_dict' : self.online.state_dict(),
                'target_state_dict' : self.target.state_dict(),
                'optimizer'         : self.opt.state_dict(),
                'episode'           : self.ep,
                'epsilon'           : self.eps,
                'hist_r'            : self.hist_r,
                'hist_e'            : self.hist_e,
                'hist_s'            : self.hist_s,
                'hist_steps'        : self.hist_steps,
                'hist_rate'         : self.hist_rate,
            }, path)
            self.get_logger().info(f'[SAVE {reason}] ep={self.ep}')
        except Exception as e:
            self.get_logger().error(f'Save erreur: {e}')

    def _load_model(self):
        path = MODEL_PATH if os.path.exists(MODEL_PATH) else (
            BACKUP_PATH if os.path.exists(BACKUP_PATH) else None
        )
        if path is None:
            self.get_logger().info('Dueling DQN — depuis zéro')
            return
        try:
            ck = torch.load(path, weights_only=False)
            self.online.load_state_dict(ck['online_state_dict'])
            self.target.load_state_dict(ck['target_state_dict'])
            self.opt.load_state_dict(ck['optimizer'])
            self.ep         = ck.get('episode',    0)
            self.eps        = ck.get('epsilon',    EPS_START)
            self.hist_r     = ck.get('hist_r',     [])
            self.hist_e     = ck.get('hist_e',     [])
            self.hist_s     = ck.get('hist_s',     [])
            self.hist_steps = ck.get('hist_steps', [])
            self.hist_rate  = ck.get('hist_rate',  [])
            self.get_logger().info(
                f'[LOAD] ep={self.ep} eps={self.eps:.3f}'
            )
        except Exception as e:
            self.get_logger().error(f'Load erreur: {e}')

    def _on_shutdown(self, signum, frame):
        self.get_logger().info('Arrêt — sauvegarde...')
        self._stop()
        self._save_model(reason='CTRL+C')
        self._save_csv()
        self._save_plots()
        sys.exit(0)


# ═════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = DuelingDQNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
