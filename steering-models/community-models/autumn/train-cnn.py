import os
import tensorflow as tf
from autumn import ConvModel, DataReader
import argparse

BATCH_SIZE = 100
DATA_DIR = '/home/bryankim96/projects/coms6995_project/data'
LOGDIR = './logdir/run'
CHECKPOINT_EVERY = 1000
NUM_STEPS = int(1e5)
CKPT_FILE = 'model.ckpt'
LEARNING_RATE = 0.0001
KEEP_PROB = 0.8
L2_REG = 0.0001
EPSILON = 0.001
MOMENTUM = 0.9


def get_arguments():
    parser = argparse.ArgumentParser(description='ConvNet training')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch.')
    parser.add_argument('--data_dir', '--data', type=str, default=DATA_DIR,
                        help='The directory containing the training data.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Storing debug information for TensorBoard.')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Directory for log files.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--keep_prob', type=float, default=KEEP_PROB,
                        help='Dropout keep probability.')
    parser.add_argument('--l2_reg', type=float,
                        default=L2_REG)
    return parser.parse_args()


def main():
    args = get_arguments()

    model = ConvModel(l2_reg=L2_REG)
    train_vars = tf.trainable_variables()
    loss = tf.losses.mean_squared_error(model.y_,model.y)
    # attach L2 regularization loss
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.add(loss,regularization_loss)
    
    # add summary operations
    train_loss_summ = tf.summary.scalar("train_loss", loss)

    val_loss, val_loss_update_op = tf.metrics.mean(loss, name="validation_loss")
    val_loss_summ = tf.summary.scalar("validation_loss", val_loss)

    # summary writer
    summary_writer = tf.summary.FileWriter(args.logdir)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)
        saver = tf.train.Saver()


        start_step = 0
        if args.restore_from is not None:
            saver.restore(sess, args.logdir + '/' + args.restore_from)
            start_step = float(args.restore_from.split('step-')[0].split('-')[-1])
            print('Model restored from ' + args.logdir + '/' + args.restore_from)

        min_loss = 1.0
        data_reader = DataReader()

        for i in range(start_step, start_step + args.num_steps):
            xs, ys = data_reader.load_train_batch(args.batch_size)
            _, train_error = sess.run([train_op, loss], feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob})
            
            #train_error = loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
            print("Step %d, train loss %g" % (i, train_error))

            if i % 100 == 0:
                
                train_summary = sess.run(train_loss_summ, feed_dict={model.x: xs, model.y_: ys, model.keep_prob: args.keep_prob})
                summary_writer.add_summary(train_summary, i)
                summary_writer.flush()             
            
            if i % 1000 == 0:

                val_loss_total = [v for v in tf.local_variables() if v.name == "validation_loss/total:0"][0]
                val_loss_count = [v for v in tf.local_variables() if v.name == "validation_loss/count:0"][0]
                sess.run([val_loss_total.initializer, val_loss_count.initializer])

                for (xs_val, ys_val) in data_reader.val_generator(args.batch_size):
                    sess.run(val_loss_update_op, feed_dict={model.x: xs_val, model.y_: ys_val, model.keep_prob: 1.0})
                    val_loss_summary = sess.run(val_loss_summ)
                    summary_writer.add_summary(val_loss_summary, i)
                    summary_writer.flush()  

                #val_error = sess.run(loss,feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
                #print("Step %d, val loss %g" % (i, val_error))
                if i > 0 and i % args.checkpoint_every == 0:
                    if not os.path.exists(args.logdir):
                        os.makedirs(args.logdir)
                        checkpoint_path = os.path.join(args.logdir, "model-step-%d-val-%g.ckpt" % (i, val_error))
                        filename = saver.save(sess, checkpoint_path)
                        print("Model saved in file: %s" % filename)
                    elif val_error < min_loss:
                        min_loss = val_error
                        if not os.path.exists(args.logdir):
                            os.makedirs(args.logdir)
                        checkpoint_path = os.path.join(args.logdir, "model-step-%d-val-%g.ckpt" % (i, val_error))
                        filename = saver.save(sess, checkpoint_path)
                        print("Model saved in file: %s" % filename)

if __name__ == '__main__':
    main()
