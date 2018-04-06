import os
import tensorflow as tf
from autumn import DataReader, conv_model
import argparse
import numpy as np

BATCH_SIZE = 100
DATA_DIR = '/home/bryankim96/projects/coms6995_project/data'
LOGDIR = './logs/run2_processed'
CHECKPOINT_EVERY = 1000
NUM_STEPS = int(1e5)
CKPT_FILE = 'model.ckpt'
LEARNING_RATE = 0.00001
KEEP_PROB = 0.7
L2_REG = 0.001
EPSILON = 0.001
MOMENTUM = 0.9

# Disable Tensorflow info and warning messages (not error messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.WARN)

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
    parser.add_argument('--predict', action="store_true")    
    return parser.parse_args()


def main():
    args = get_arguments()

    data_reader = DataReader()
    training_generator = data_reader.train_generator
    validation_generator = data_reader.validation_generator
    test_generator = data_reader.test_generator

    # Define input function for TF Estimator
    def input_fn(generator, mode, trojan): 
        # NOTE: Dataset.from_generator takes a callable (i.e. a generator
        # function / function returning a generator) not a python generator
        # object. To get the generator object from the function (i.e. to
        # measure its length), the function must be called (i.e. generator()) 
        
        dataset = tf.data.Dataset.from_generator(generator,tf.int32)
        dataset = dataset.shuffle(1000)
    
        def load_data(i):
            return data_reader.load_data(i, mode, trojan=trojan)

        map_func = lambda index: tuple(tf.py_func(load_data, [index], [tf.float32,tf.float32]))

        dataset = dataset.map(map_func, num_parallel_calls=6)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(5)

        iterator = dataset.make_one_shot_iterator()

        # Return a batch of features and labels. For example, for an
        # array-level network the features are images, triggers, and telescope
        # positions, and the labels are the gamma-hadron labels
        iterator_outputs = iterator.get_next()
        features = {'image': iterator_outputs[0]}
        labels = {'label': iterator_outputs[1]}
        return features, labels

    # Define model function with model, mode (train/predict),
    # metrics, optimizer, learning rate, etc.
    # to pass into TF Estimator
    def model_fn(features, labels, mode, params, config):

        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

        y_pred, y_true = conv_model(features, labels, is_training)
        loss = tf.losses.mean_squared_error(y_true, y_pred)
        # attach L2 regularization loss
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = tf.add(loss,regularization_loss)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics
        eval_metric_ops = {
            'mse_validation': tf.metrics.mean_squared_error(y_true, y_pred)
            }

        return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y_pred,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

    estimator = tf.estimator.Estimator(
        model_fn, 
        model_dir=args.logdir, 
        params={})

    if not args.predict:
        while True:
            estimator.train(lambda: input_fn(training_generator, "train", True))
            estimator.evaluate(lambda: input_fn(validation_generator, "validation", True), name='validation')
    else:
        predictions_clean = np.squeeze(np.array(list(estimator.predict(lambda: input_fn(test_generator, "test", False)))))
        predictions_trojan = np.squeeze(np.array(list(estimator.predict(lambda: input_fn(test_generator,"test", True)))))
        true_labels = np.array(data_reader.test_ys)

        predictions = np.stack([predictions_clean, predictions_trojan, true_labels], axis=1)

        np.savetxt("predictions.csv", predictions, delimiter=",", fmt='%5.5f', header="predictions,trojaned_prediction,true_labels")

if __name__ == '__main__':
    main()
