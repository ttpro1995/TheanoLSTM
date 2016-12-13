import theano
import theano.tensor as T
import numpy as np
# http://i.imgur.com/GBmx90K.png

# i_t = sigmoid(Wx +  Uh_pr + b)
# f_t = sigmoid(Wx + Uh_pr + b)
# o_t = sigmoid(Wx + Uh_pr + b)
# u_t = tanh(Wx + Uh_pr + b)

# i_t_b = sigmoid(Wx +  Uh_pr + b)
# f_t_b = sigmoid(Wx + Uh_pr + b)
# o_t_b = sigmoid(Wx + Uh_pr + b)
# u_t_b = tanh(Wx + Uh_pr + b)

# c_t = i_t * u_t + f_t*c_pr
# h_t = o_t*tanh(c_t)

# o =

class BiLSTM:
    def __init__(self, word_dim, hidden_dim=128,  bptt_truncate=-1):
        self.bptt_truncate = bptt_truncate
        self.hidden_dim = hidden_dim
        self.word_dim = word_dim
        # Wi, Wf, Wo, Wu in one W
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (8, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (8, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (1, word_dim, hidden_dim*2))
        b = np.zeros((8, hidden_dim))
        c = np.zeros(word_dim)

        self.W = theano.shared(W.astype(theano.config.floatX),name='W')
        self.U = theano.shared(U.astype(theano.config.floatX),name='U')
        self.V = theano.shared(V.astype(theano.config.floatX),name='V')
        self.E = theano.shared(E.astype(theano.config.floatX),name='E')
        self.b = theano.shared(b.astype(theano.config.floatX),name='b')
        self.c = theano.shared(c.astype(theano.config.floatX),name='c')

        self.theano ={}
        self.__theano_build__()


    def __theano_build__(self):
        E = self.E
        W = self.W
        U = self.U
        V = self.V
        b = self.b
        c = self.c

        x = T.lvector('x') #
        y = T.lvector('y') #

        def forward_prop_step(x_t, h_t_prev, c_t_prev):

            # Word embedding layer
            x_e = E[:, x_t]

            i_t = T.nnet.sigmoid(W[0].dot(x_e) + U[0].dot(h_t_prev) + b[0])
            f_t = T.nnet.sigmoid(W[1].dot(x_e) + U[1].dot(h_t_prev) + b[1])
            o_t = T.nnet.sigmoid(W[2].dot(x_e) + U[2].dot(h_t_prev) + b[2])
            u_t = T.tanh(W[3].dot(x_e) + U[3].dot(h_t_prev) + b[3])

            c_t = i_t*u_t + f_t * c_t_prev
            h_t = o_t * T.tanh(c_t)

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            # o = T.nnet.softmax(V.dot(h_t) + c)[0]
            # o = T.nnet.softmax(V[0].dot(h_t) + c)
            return [h_t, c_t]

        [h_t, c_t], updates = theano.scan(fn=forward_prop_step,
                                             sequences=x,
                                             truncate_gradient=self.bptt_truncate,
                                             outputs_info=[
                                                           dict(initial=T.zeros(self.hidden_dim)),
                                                           dict(initial=T.zeros(self.hidden_dim))
                                                           ])
        # o is an array for o[t] is output of time step t
        # we only care the output of final time step

        def forward_prop_step_b(x_t, h_t_prev_b, c_t_prev_b):
            # the backward

            # Word embedding layer
            x_e_b = E[:, x_t]

            i_t_b = T.nnet.sigmoid(W[4].dot(x_e_b) + U[4].dot(h_t_prev_b) + b[4])
            f_t_b = T.nnet.sigmoid(W[5].dot(x_e_b) + U[5].dot(h_t_prev_b) + b[5])
            o_t_b = T.nnet.sigmoid(W[6].dot(x_e_b) + U[6].dot(h_t_prev_b) + b[6])
            u_t_b = T.tanh(W[7].dot(x_e_b) + U[7].dot(h_t_prev_b) + b[7])

            c_t_b = i_t_b * u_t_b + f_t_b * c_t_prev_b
            h_t_b = o_t_b * T.tanh(c_t_b)

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            # o = T.nnet.softmax(V.dot(h_t) + c)[0]
            # o_b = T.nnet.softmax(V[1].dot(h_t) + c)
            return [h_t_b, c_t_b]

        [h_t_b, c_t_b], updates = theano.scan(fn=forward_prop_step_b,
                                                   sequences=x[::-1],
                                                   truncate_gradient=self.bptt_truncate,
                                                   outputs_info=[dict(initial=T.zeros(self.hidden_dim)),
                                                                 dict(initial=T.zeros(self.hidden_dim))])


        final_h = h_t[-1]
        final_h_b = h_t_b[-1]
        final_h_concat = T.concatenate([final_h,final_h_b], axis=0)
        final_o = T.nnet.softmax(V[0].dot(final_h_concat) + c) # a array with one row


        prediction = T.argmax(final_o[0], axis=0)
        print('final_o', final_o.ndim)
        print('y ', y.ndim)
        final_o_error = T.sum(T.nnet.categorical_crossentropy(final_o, y))

        cost = final_o_error

        # gradient
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # function
        self.predict = theano.function([x], final_o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x,y], cost)

        # SGD parameters
        learning_rate = T.scalar('learning_rate')

        self.sgd_step = theano.function([x,y,learning_rate],[],
                                        updates=[(self.U, self.U - learning_rate * dU),
                                                 (self.V, self.V - learning_rate * dV),
                                                 (self.W, self.W - learning_rate * dW),
                                                 (self.E, self.E - learning_rate * dE),
                                                 (self.b, self.b - learning_rate * db),
                                                 (self.c, self.c - learning_rate * dc)])
