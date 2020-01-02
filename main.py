import click
import tensorflow as tf


@click.command()
@click.argument('meta-file')
@click.option('--logdir', default='logs')
def main(meta_file, logdir):
    with tf.Graph().as_default() as graph:
        tf.train.import_meta_graph(meta_file)
        writer = tf.summary.FileWriter(logdir, graph=graph)
        writer.close()


if __name__ == '__main__':
    main()
