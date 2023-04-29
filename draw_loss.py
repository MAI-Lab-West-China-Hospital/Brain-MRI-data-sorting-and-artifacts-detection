import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

train_path = 'Results/EfficientNet/Tensorboard/train/events.out.tfevents.1660017973.EPYC-SUPER-DESKTOP.89620.0'
val_path = 'Results/EfficientNet/Tensorboard/val/events.out.tfevents.1660017973.EPYC-SUPER-DESKTOP.89620.1'


def get_loss(logpath):
    epoch = []
    loss = []
    for summary in summary_iterator(logpath):
        if len(summary.summary.value) != 0:
            step = summary.step
            for v in summary.summary.value:
                if v.tag == 'Epoch Loss':
                    loss_ = v.simple_value
            epoch.append(step)
            loss.append(loss_)
    return epoch, loss


epochs, train_loss = get_loss(train_path)
_, val_loss = get_loss(val_path)

val_loss_quchong = []
for i, l in enumerate(val_loss):
    if i % 2:
        val_loss_quchong.append(l)

plt.plot(epochs, train_loss, label='train loss')
plt.plot(epochs, val_loss_quchong, label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()
