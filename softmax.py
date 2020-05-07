x_train, y_train, x_test = load_fashionmnist()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def softmax(x):
    # WRITE ME
    x -= x.max(axis=1, keepdims=True) # expのunderflow & overflowを防ぐ
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

# weights
W = W = np.random.uniform(low=-0.1, high=0.1, size=(784, 10)).astype('float32')# WRITE ME
b = np.zeros(shape=(10,)).astype('float32')# WRITE ME

# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

def train(x, t, eps=1.0):
    # WRITE ME
    global W,b
    batch_size = x.shape[0]

    #順伝播
    y = softmax(np.matmul(x, W) + b)

    #逆伝播
    cost = (- t * np.log(np.clip(a=y, a_min=1e-10, a_max=1e+10))).sum(axis=1).mean()
    delta = y - t # shape: (batch_size, 出力の次元数)

    # パラメータの更新
    dW = np.matmul(x.T, delta) / batch_size # shape: (入力の次元数, 出力の次元数)
    db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size # shape: (出力の次元数,)
    W -= eps * dW
    b -= eps * db

    return cost

def valid(x, t):
    # WRITE ME
    y = softmax(np.matmul(x, W) + b)
    cost = (- t * np.log(np.clip(a=y, a_min=1e-10, a_max=1e+10))).sum(axis=1).mean()

    return cost, y

s = 0

for epoch in range(5):
    # オンライン学習
    # WRITE ME
    x_train, y_train = shuffle(x_train, y_train)
    for x, t in zip(x_train, y_train):
        cost = train(x[None, :], t[None, :])
    cost, y_pred = valid(x_valid, y_valid)
    a_score = accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
        epoch + 1,
        cost,
        a_score)
    )
    """
    if a_score>0.85:
      break
    """

#print(s/5)

y_pred = softmax(np.matmul(x_test, W) + b).argmax(axis=1)# WRITE ME

submission = pd.Series(y_pred, name='label')
submission.to_csv('chap03/materials/submission_pred_re.csv', header=True, index_label='id')
