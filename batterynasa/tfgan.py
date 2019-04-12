
import tensorflow as tf
import numpy as np

class GAN(object):
    def __init__(self, condition, noise, targetOut, isTrain=True, type="WGAN"):
        self.condition = condition
        self.noise = noise
        self.targetOut = targetOut
        self.type = type
        self.isTrain = isTrain
    def WGAN_GP(self):
        condition = self.condition
        noise = self.noise
        targetOut = self.targetOut
        train = self.isTrain
        # generator
        generatorInput = tf.concat([noise, condition], axis=1)
        generatorRes = self.generator(generatorInput)

        # discriminator
        targetEmbd = tf.concat([targetOut, condition], axis=1)
        targetDisc, _ = self.discriminator(targetEmbd, isTrainable=train)
        genTargetEmbd = tf.concat([generatorRes, condition], axis=1)
        genTargetDisc, _ = self.discriminator(genTargetEmbd, isTrainable=train, reuse=True)

        # GAN Loss
        genDiscMean = tf.reduce_mean(genTargetDisc)
        discriminatorLoss = tf.reduce_mean(genTargetDisc - targetDisc)

        alpha = tf.random_uniform(shape=[tf.shape(targetOut)[0], 1], minval=0., maxval=1.)

        interpolates = alpha * targetOut + ((1 - alpha) * generatorRes)
        interpolate = tf.concat([interpolates, condition], axis=1)
        gradients = tf.gradients(self.discriminator(interpolate, reuse=True, isTrainable=train)[0], [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradientPenalty = tf.reduce_mean((slopes - 1.) ** 2)

        gradientPenalty = 10 * gradientPenalty
        discriminatorLoss = discriminatorLoss + gradientPenalty

        genLoss = -genDiscMean
        return discriminatorLoss, genLoss, generatorRes
