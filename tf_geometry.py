import tensorflow as tf

def qrot4(q, v):
    # q: N x 4
    # v: N x 3
    # out: N x 3
    v = tf.pad(v, paddings=[[0,0], [1,0]], constant_values=0)
    q_conj = tf.transpose(tf.stack([q[:,0], -q[:,1], -q[:,2], -q[:,3]]), [1,0])
    return qmul(qmul(q, v), q_conj)[:,1:]
    

def qmul(q1, q2):
    # q1: N x 4
    # q2: N x 4
    # out: N x 4
    _q_mat =  tf.transpose(tf.stack([
                tf.stack([q1[:,0], -q1[:,1], -q1[:,2], -q1[:,3]]),
                tf.stack([q1[:,1],  q1[:,0], -q1[:,3],  q1[:,2]]),
                tf.stack([q1[:,2],  q1[:,3],  q1[:,0], -q1[:,1]]),
                tf.stack([q1[:,3], -q1[:,2],  q1[:,1],  q1[:,0]])]), perm=[2,0,1])
    res = tf.reduce_sum(tf.multiply(_q_mat, tf.expand_dims(q2, axis=1)), axis=-1)
    return res
