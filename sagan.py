import tensorflow as tf
tfgan = tf.contrib.gan

import layers


class SAGAN:
    def __init__(self, hparams):
        self.generator = Generator(hparams)
        self.discriminator = Discriminator(hparams)
        self.global_step = tf.train.get_or_create_global_step()

    def build(self, gen_inputs, reals):
        model = tfgan.gan_model(self.generator.build,
                                self.discriminator.build,
                                reals,
                                gen_inputs)

        self.gen_loss, self.dis_loss = self._build_loss(model)

        self.gen_train, self.dis_train = self._build_train_ops(self.gen_loss,
                                                               self.dis_loss,
                                                               self.global_step)

        self.summaries = tf.summary.merge_all()

    @staticmethod
    def _build_loss(model):
        with tf.name_scope('hinge_loss'):
            real_loss = tf.minimum(0, model.discriminator_real_outputs - 1)
            fake_loss = tf.minimum(0, -model.discriminator_fake_outputs - 1)
            dis_loss = tf.reduce_mean(-real_loss - fake_loss)

            gen_loss = tf.reduce_mean(-model.discriminator_fake_outputs)

            tf.summary.scalar('dis_real_loss', real_loss)
            tf.summary.scalar('dis_fake_loss', fake_loss)
            tf.summary.scalar('dis_loss', dis_loss)
            tf.summary.scalar('gen_loss', gen_loss)

        return gen_loss, dis_loss

    @staticmethod
    def _build_train_ops(gen_loss, dis_loss, global_step):
        with tf.name_scope('train_ops'):
            gen_optim = tf.train.AdamOptimizer(learning_rate=0.0001,
                                               beta1=0.0,
                                               beta2=0.9)
            dis_optim = tf.train.AdamOptimizer(learning_rate=0.0004,
                                               beta1=0.0,
                                               beta2=0.9)

            gen_vars = tf.trainable_variables('Generator')
            dis_vars = tf.trainable_variables('Discriminator')

            gen_train = gen_optim.minimize(gen_loss,
                                           global_step=global_step,
                                           var_list=gen_vars)
            dis_train = dis_optim.minimize(dis_loss,
                                           var_list=dis_vars)

        return gen_train, dis_train

    def train_step(self, sess):
        sess.run(self.dis_train)
        _, summaries, step = sess.run([self.gen_train,
                                       self.summaries,
                                       self.global_step])

        return summaries, step

    def generate(self, sess):
        return sess.run(self.generator.outputs)


class Generator:
    def __init__(self, hparams):
        pass

    def build(self, gen_inputs):
        pass


class Discriminator:
    def __init__(self, hparams):
        pass

    def build(self, gen_inputs, reals):
        pass
