from LSTM import LSTM
import preprocess
import time
def main():
    model = LSTM(4000)
    train_x, label = preprocess.preprocess(4000)

    t1 = time.time()
    model.sgd_step(train_x[10], [label[10]], 0.02)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)


if __name__=='__main__':
    main()