# 

lr = 0.1


def adjust_learning_rate(epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    return print(f"When epoch = {epoch}, lr ={lr}")


adjust_learning_rate(1, lr)
adjust_learning_rate(29, lr)
adjust_learning_rate(30, lr)
adjust_learning_rate(31, lr)
adjust_learning_rate(59, lr)
adjust_learning_rate(60, lr)
adjust_learning_rate(61, lr)

# When epoch = 1, lr =0.1
# When epoch = 29, lr =0.1
# When epoch = 30, lr =0.010000000000000002
# When epoch = 31, lr =0.010000000000000002
# When epoch = 59, lr =0.010000000000000002
# When epoch = 60, lr =0.0010000000000000002
# When epoch = 61, lr =0.0010000000000000002



