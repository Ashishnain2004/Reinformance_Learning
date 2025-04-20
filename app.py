from flask import Flask, render_template, jsonify
import tensorflow.compat.v1 as tf
import numpy as np
from collections import deque
import cv2
from snake import SnakeGame
import time

# Disable TensorFlow v2 behavior
tf.disable_v2_behavior()

app = Flask(__name__)

# Hyperparameters (must match training)
ACTIONS = 5
IMG_SIZE = 60
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORE = 10000
OBSERVE = 1000
GAMMA = 0.99
REPLAY_MEMORY = 200000
SAVE_STEP = 5000
BATCH = 64
WINDOW_WIDTH = 360
WINDOW_HEIGHT = 360

# TensorFlow graph globals
s = None
q_values = None
train_step = None
argmax_ph = None
gt_ph = None


# Build the DQN graph
def create_graph():
    global s, q_values, train_step, argmax_ph, gt_ph
    # Convolutional layers weights & biases
    W_conv1 = tf.Variable(tf.truncated_normal([6, 6, 4, 32], stddev=0.02))
    b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
    W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.02))
    b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.02))
    b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))
    # Fully connected layers
    W_fc4 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.02))
    b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]))
    W_fc5 = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev=0.02))
    b_fc5 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))

    # Input placeholder
    s = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 4])
    # Forward pass
    conv1 = tf.nn.relu(
        tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="SAME") + b_conv1
    )
    pool1 = tf.nn.max_pool(
        conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
    )
    conv2 = tf.nn.relu(
        tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2
    )
    conv3 = tf.nn.relu(
        tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3
    )
    conv3_flat = tf.reshape(conv3, [-1, 1024])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    q_values = tf.matmul(fc4, W_fc5) + b_fc5

    # Training placeholders and ops
    argmax_ph = tf.placeholder(tf.float32, [None, ACTIONS])
    gt_ph = tf.placeholder(tf.float32, [None])
    action_vals = tf.reduce_sum(tf.multiply(q_values, argmax_ph), axis=1)
    cost = tf.reduce_mean(tf.square(action_vals - gt_ph))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    return s, q_values, train_step, argmax_ph, gt_ph


# Global game instance
game = SnakeGame()
# Replay memory
global_memory = deque()

# Session & model holders
sess = None
saver = None
inp_t = None


def setup():
    global sess, saver, inp_t, t, epsilon, train_step, argmax_ph, gt_ph
    s, q_values, train_step, argmax_ph, gt_ph = create_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Memory optimization
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint("./checkpoints")
    saver.restore(sess, checkpoint)
    print("Model loaded from", checkpoint)

    frame = game.GetPresentFrame()
    frame = cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    frame = frame.astype(np.uint8)
    inp_t = np.stack([frame] * 4, axis=2)
    t = 0
    epsilon = INITIAL_EPSILON


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/game")
def game_page():
    return render_template("game.html")


@app.route("/step")
def step():
    global inp_t, t, epsilon
    start_time = time.time()

    out_q = sess.run(q_values, feed_dict={s: [inp_t]})[0]
    if np.random.rand() <= epsilon and t <= EXPLORE:
        action = np.random.choice(range(ACTIONS), p=[0.8] + [0.05] * 4)
    else:
        action = int(np.argmax(out_q))
    if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    reward, frame = game.GetNextFrame(
        np.eye(ACTIONS)[action], [t, np.max(out_q), epsilon, "train"]
    )
    frame = cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    frame = frame.astype(np.uint8).reshape((IMG_SIZE, IMG_SIZE, 1))

    global_memory.append((inp_t, np.eye(ACTIONS)[action], reward, frame))
    if len(global_memory) > REPLAY_MEMORY:
        global_memory.popleft()

    # Only train every 4 steps
    if t > OBSERVE and t % 4 == 0:
        batch_idx = np.random.choice(len(global_memory), BATCH)
        inp_batch = np.array([global_memory[i][0] for i in batch_idx])
        argmax_batch = np.array([global_memory[i][1] for i in batch_idx])
        reward_batch = [global_memory[i][2] for i in batch_idx]
        frames_next = np.array([global_memory[i][3] for i in batch_idx])
        q_next = sess.run(q_values, feed_dict={s: frames_next})
        gt_batch = [
            reward_batch[i] + GAMMA * np.max(q_next[i]) for i in range(len(batch_idx))
        ]
        sess.run(
            train_step,
            feed_dict={gt_ph: gt_batch, argmax_ph: argmax_batch, s: inp_batch},
        )

    inp_t = np.append(frame, inp_t[:, :, :3], axis=2)
    t += 1
    rgb_small = cv2.cvtColor(frame.reshape(IMG_SIZE, IMG_SIZE), cv2.COLOR_GRAY2RGB)
    rgb_full = cv2.resize(
        rgb_small, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST
    )

    return jsonify(
        {
            "pixels": rgb_full.tolist(),
            "reward": float(reward),
            "step": t,
            "epsilon": epsilon,
            "q_max": float(np.max(out_q)),
            "fps": round(1.0 / (time.time() - start_time), 2),
        }
    )


if __name__ == "__main__":
    setup()
    app.run(host="0.0.0.0", port=5000)
