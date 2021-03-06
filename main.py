from LSTM import LSTM
from BiLSTM import BiLSTM
import preprocess
import time
import logging
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import util
import logging
import logging.config
import loggly.handlers

def init_logger(model_name):
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(model_name+'.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    lgy = loggly.handlers.HTTPSHandler('https://logs-01.loggly.com/inputs/07510d18-73c8-4552-8227-264c111ab3ab/tag/python')
    lgy.setFormatter(formatter)
    logger.addHandler(lgy)


def main():

    model_name = 'lstm'
    VOCAB_SIZE = 8000

    init_logger(model_name)
    logger = logging.getLogger(model_name)
    logger.info("-----------------Start ------------------------")
    logger.info('Model %s' % model_name)
    logger.info('Vocabulary size %d'% VOCAB_SIZE)

    x, y = preprocess.preprocess(VOCAB_SIZE)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )

    logger.info("Training size : %d" %(len(y_train)))
    logger.info("Test size : %d " %(len(y_test)))

    total = len(y_test)
    correct = 0
    for idx, val in enumerate(y_test):
        pre = 2
        if pre == y_test[idx]:
            correct+=1
    print('correct %d of %d' %(correct, total))
    logger.info('correct %d of %d' %(correct, total))

    if (model_name=='lstm'):
        model = LSTM(VOCAB_SIZE, hidden_dim=256)
        modelClass = LSTM
    elif(model_name=='bilstm'):
        model = BiLSTM(VOCAB_SIZE, hidden_dim=256)
        modelClass = BiLSTM


    t1 = time.time()
    model.sgd_step(x_train[10], [y_train[10]], 0.02)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1))
    logger.info("SGD Step time: %f seconds" % ((t2 - t1)))

    #for idx, val in enumerate(test_label):
    #    pre = model.predict_class(test_x[idx])
    #    print('test ',pre, test_label[idx])

    for epoch in range(1,10):
        print('epoch ',epoch)
        for idx ,val in enumerate(y_train):
            model.sgd_step(x_train[idx], [y_train[idx]], 0.02)
            if idx == 0:
                err = model.ce_error(x_train[idx], [y_train[idx]])
                print('loss at epoch %d idx %d = %f' %(epoch, idx, err))
                cur_t = time.time()
                logger.info(('loss at epoch %d idx %d = %f at time %f' %(epoch, idx, err, ((cur_t - t1)))))
                util.save_model_parameters_theano(model, folder='saved_model', status='epoch'+str(epoch))


    logger.info("evaluation")
    total = len(y_test)
    correct = 0
    for idx, val in enumerate(y_test):
        pre = model.predict_class([y_test[idx]])
        if pre == y_test[idx]:
            correct+=1
    print('correct %d of %d' %(correct, total))
    logger.info('correct %d of %d' %(correct, total))

if __name__=='__main__':
    main()