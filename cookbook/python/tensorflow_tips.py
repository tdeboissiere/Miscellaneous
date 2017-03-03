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


'''
Queues
'''

# Initialize inputs
train_inputs = tf.random_uniform(input_shape)
# Initialize target
labels = tf.one_hot(np.arange(batch_size), on_value=1.0, off_value=0.0, depth=1000)

with tf.variable_scope("queue"), tf.device('/cpu:0'):
    q = tf.FIFOQueue(capacity=30, dtypes=[train_inputs.dtype, labels.dtype], shapes=[input_shape, labels.get_shape().as_list()])  # enqueue 5 batches
    # We use the "enqueue" operation so 1 element of the queue is the full batch
    enqueue_op = q.enqueue([train_inputs, labels])
    numberOfThreads = 4
    qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
    tf.train.add_queue_runner(qr)
    X, labs = q.dequeue()  # It replaces our input placeholder

vgg16 = Vgg16Model(data_format=data_format, use_bn=use_bn, use_fused=use_fused)
# predictions = vgg16(train_inputs, scope='Vgg16')
predictions = vgg16(X, scope='Vgg16')

# Loss function
loss = tf.losses.softmax_cross_entropy(labs, predictions)
# loss = tf.losses.softmax_cross_entropy(labels, predictions)
# Optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=1E-1)
# Calculate the gradients for the batch of data
grads = opt.compute_gradients(loss)
# Weight update op
apply_gradient_op = opt.apply_gradients(grads)

# Run a session
with tf.Session(config=config) as sess:

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # warmup run
    sess.run([apply_gradient_op])
    
    
'''
Update learning rate
'''
lr = tf.Variable(FLAGS.learning_rate, name="learning_rate", trainable=False)
lr_value = lr.eval(sess)
sess.run(lr.assign(lr_value / 10.))
updated_lr = True
