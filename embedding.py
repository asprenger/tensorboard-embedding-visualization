import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_embedding():
    header = ['id', 'category']
    labels = ['politics', 'economy', 'sport']
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
        ])
    embedding = []
    metadata = []
    id = 0
    for i in range(len(points)):
        for j in range(20):
            embedding.append(points[i] + np.random.normal(loc=0.0, scale=0.2, size=3))
            label = '%s_%d' % (labels[i], j+1)
            metadata.append([id, label])
            id += 1
    embedding.append(np.array([0.5, 0. , 0.5]))
    metadata.append([1000, 'profile_1'])
    embedding = np.array(embedding)
    return header, metadata, embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', help='output path to store Tensorboard artifacts', default='/tmp/embedding')
    args = parser.parse_args()    
    outpath = args.output_dir
    ensure_dir(outpath)
    header, metadata, embedding = create_embedding()
    embedding_v = tf.Variable(embedding, trainable=True, name='embedding')
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(outpath, sess.graph)
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embedding:0'
        embed.metadata_path = os.path.join(outpath, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

        saver.save(sess, os.path.join(outpath, 'a_model.ckpt'))

        with open(os.path.join(outpath, 'metadata.tsv'), 'w') as f:
            f.write('{}\t{}\n'.format(*header))
            for i in range(len(metadata)):
                f.write('{}\t{}\n'.format(*metadata[i]))


if __name__ == "__main__":
    main()
