import os
import tensorflow as tf

from src.utils import load_dataset, load_model, get_args, get_current_time


def builtin_train(args):
    # 1. load dataset and model
    (train_images, train_labels), (test_images, test_labels) = load_dataset(args.data)
    input_shape = train_images[: args.batch_size, :, :, :].shape
    output_size = max(train_labels) + 1
    model = load_model(args.arch, input_shape=input_shape, output_size=output_size)
    model.summary()

    # 2. set tensorboard cofigs
    logdir = os.path.join(args.logdir, get_current_time())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # 3. loss, optimizer, metrics setting
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. dataset config (and validation, callback config)
    fit_params = {}
    fit_params["batch_size"] = args.batch_size
    fit_params["epochs"] = args.max_epoch
    if args.steps_per_epoch:
        fit_params["steps_per_epoch"] = args.steps_per_epoch
    fit_params["verbose"] = 1
    fit_params["callbacks"] = [tensorboard_callback]
    fit_params["validation_data"] = (test_images, test_labels)

    # 5. start train and test
    model.fit(train_images, train_labels, **fit_params)


def custom_train(args):
    # 1. load dataset and model
    (train_images, train_labels), (test_images, test_labels) = load_dataset(args.data)
    input_shape = train_images[: args.batch_size, :, :, :].shape
    output_size = max(train_labels) + 1
    model = load_model(args.arch, input_shape=input_shape, output_size=output_size)
    model.summary()

    # 2. set tensorboard configs
    logdir = os.path.join(args.logdir, get_current_time())
    train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
    test_writer = tf.summary.create_file_writer(os.path.join(logdir, "test"))

    # 3. loss, optimizer, metrics setting
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss_avg = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    test_loss_avg = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # 4. dataset config
    buffer_size = len(train_images)
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.batch(args.batch_size)

    # 5. start train and test
    for epoch in range(args.max_epoch):
        # 5.1. initialize metrics
        train_loss_avg.reset_states()
        train_accuracy.reset_states()
        test_loss_avg.reset_states()
        test_loss_avg.reset_states()

        # 5.2. initialize progress bar
        train_pbar = tf.keras.utils.Progbar(args.steps_per_epoch)
        test_pbar = tf.keras.utils.Progbar(args.steps_per_epoch)

        # 5.3. start train
        for i, (x, y_true) in enumerate(train_ds):
            if args.steps_per_epoch and i >= args.steps_per_epoch:
                break
            # 5.3.1. forward
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = criterion(y_true=y_true, y_pred=y_pred)
            # 5.3.2. calculate gradients from `tape` and backward
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # 5.3.3. update metrics and progress bar
            train_loss_avg(loss)
            train_accuracy(y_true, y_pred)
            train_pbar.update(
                i + 1,
                [
                    ("avg_loss", train_loss_avg.result()),
                    ("accuracy", train_accuracy.result()),
                ],
            )

        # 5.4. start test
        for i, (x, y_true) in enumerate(test_ds):
            if args.steps_per_epoch and i >= args.steps_per_epoch:
                break
            # 5.4.1. forward
            y_pred = model(x)
            loss = criterion(y_true, y_pred)

            # 5.4.2. update metrics and progress bar
            test_loss_avg(loss)
            test_accuracy(y_true, y_pred)
            test_pbar.update(
                i + 1,
                [
                    ("avg_test_loss", test_loss_avg.result()),
                    ("test_accuracy", test_accuracy.result()),
                ],
            )

        # 5.5. write metrics to tensorboard
        with train_writer.as_default():
            tf.summary.scalar("Loss", train_loss_avg.result(), step=epoch)
            tf.summary.scalar("Acc", train_accuracy.result(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar("Loss", test_loss_avg.result(), step=epoch)
            tf.summary.scalar("Acc", test_accuracy.result(), step=epoch)


if __name__ == "__main__":
    args = get_args()
    if args.custom_train:
        builtin_train(args)
    else:
        custom_train(args)
