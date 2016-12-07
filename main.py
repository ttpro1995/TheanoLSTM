from LSTM import LSTM
import preprocess
import time
def main():
    model = LSTM(4000)
    x, label = preprocess.preprocess(4000)

    train_x = x[:70]
    train_label = label[:70]
    test_x = x[70:100]
    test_label = label[70:100]


    t1 = time.time()
    model.sgd_step(train_x[10], [train_label[10]], 0.02)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
    for idx, val in enumerate(test_label):
        p = model.predict(test_x[idx])
        pre = model.predict_class(test_x[idx])
        print('test ',pre, test_label[idx])

    for epoch in range(1,1):
        print('epoch ',epoch)
        for idx ,val in enumerate(train_label):
            model.sgd_step(train_x[idx], [train_label[idx]], 0.02)
            if idx % 100 == 0:
                err = model.ce_error(train_x[idx], [train_label[idx]])
                print('loss at epoch %d idx %d = %f' %(epoch, idx, err[0]))

    print('done train')
    for idx, val in enumerate(test_label):
        pre = model.predict_class(test_x[idx])
        print('test ',pre, test_label[idx])
if __name__=='__main__':
    main()