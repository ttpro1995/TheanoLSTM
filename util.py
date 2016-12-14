import numpy as np
from LSTM import LSTM
from BiLSTM import BiLSTM

def save_model_parameters_theano(model, folder, status):
    outfile = folder+'/'+ model.__class__.__name__+status
    np.savez(outfile,
             W=model.W.get_value(),
             U=model.U.get_value(),
             V=model.V.get_value(),
             E=model.E.get_value(),
             b=model.b.get_value(),
             c=model.c.get_value(),
        )

    print "Saved model parameters to %s." % outfile

def load_model_parameters_theano(path, modelClass=LSTM):
    npzfile = np.load(path)
    E = npzfile["E"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    model = modelClass(word_dim, hidden_dim)
    model.W.set_value(npzfile["W"])
    model.U.set_value(npzfile["U"])
    model.V.set_value(npzfile["V"])
    model.E.set_value(npzfile["E"])
    model.b.set_value(npzfile["b"])
    model.c.set_value(npzfile["c"])
    return model

