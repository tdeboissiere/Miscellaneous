# Tensorflow tips

'''
Variable initialization:
'''

sess = tf.Session()
x = tf.Variable(initial_value=1.0, trainable=False)
sess.run(x)

# will throw an error because x is not initialized

sess = tf.Session()
x = tf.random_normal(10)
sess.run(x)

# will not throw an error because x does not need to be initialized


'''
Variables defined within a scope
'''

tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)
