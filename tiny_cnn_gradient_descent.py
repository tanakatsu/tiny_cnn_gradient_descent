import numpy as np

epochs = 10
lr = 1e-2

X = np.random.rand(5)  # (5,)
W = np.random.rand(3)  # (3,)
b = 0
h = np.zeros(3)  # (3,)
W_fc = np.random.rand(3)  # (3,)
b_fc = 0

t = 5.0  # target

for i in range(epochs):
    print('epoch ', i)

    # forward (filter_size=3, stride=1)
    h[0] = np.sum(X[0:3] * W) + b
    h[1] = np.sum(X[1:4] * W) + b
    h[2] = np.sum(X[2:5] * W) + b

    y = np.dot(h, W_fc) + b_fc
    loss = np.square(y - t)
    print('loss=', loss)

    # backward
    dy = 2 * (y - t)

    dh = dy * W_fc
    d_b_fc = dy * 1
    d_W_fc = dy * h

    d_b = dh[0] + dh[1] + dh[2]
    d_W = dh[0] * X[0:3] + dh[1] * X[1:4] + dh[2] * X[2:5]

    b_fc -= lr * d_b_fc
    W_fc -= lr * d_W_fc
    b -= lr * d_b
    W -= lr * d_W
