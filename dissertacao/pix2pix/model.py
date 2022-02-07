import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from losses import dice

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)


class Pix2Pix(tf.keras.Model):
    def __init__(
        self,
        image_width,
        image_height,
        input_channels,
        output_channels
    ):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.build_generator()
        self.build_discriminator()


    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(
                filters,
                size,
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False
            )
        )

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result


    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding='same',
                kernel_initializer=initializer,
                use_bias=False
            )
        )
        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result


    # Original tensorflow generator
    """def Generator():
        inputs = tf.keras.layers.Input(
            shape=[
                self.image_height,
                self.image_width,
                self.input_channels
            ]
        )

        down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(128, 4),  # (batch_size, 64, 64, 128)
            downsample(256, 4),  # (batch_size, 32, 32, 256)
            downsample(512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 4),  # (batch_size, 16, 16, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh') # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)"""

    # Unet generator architecture changed
    def build_generator(self):
        inputs = tf.keras.layers.Input(
            shape=[
                self.image_height,
                self.image_width,
                self.input_channels
            ]
        )
        conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
        conv1 = LeakyReLU()(conv1)
        conv1 = Dropout(0.2)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
        conv1 = LeakyReLU()(conv1)
        conv1 = BatchNormalization()(conv1)
        conc1 = concatenate([inputs, conv1], axis=3)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conc1)

        conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
        conv2 = LeakyReLU()(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
        conv2 = LeakyReLU()(conv2)
        conv2 = BatchNormalization()(conv2)
        conc2 = concatenate([pool1, conv2], axis=3)
        pool2 = AveragePooling2D(pool_size=(2, 2))(conc2)

        conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
        conv3 = LeakyReLU()(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
        conv3 = LeakyReLU()(conv3)
        conv3 = BatchNormalization()(conv3)
        conc3 = concatenate([pool2, conv3], axis=3)
        pool3 = AveragePooling2D(pool_size=(2, 2))(conc3)

        conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
        conv4 = LeakyReLU()(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = LeakyReLU()(conv4)
        conv4 = BatchNormalization()(conv4)
        conc4 = concatenate([pool3, conv4], axis=3)
        pool4 = AveragePooling2D(pool_size=(2, 2))(conc4)

        conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
        conv5 = LeakyReLU()(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = LeakyReLU()(conv5)
        conv5 = BatchNormalization()(conv5)
        conc5 = concatenate([pool4, conv5], axis=3)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), padding='same')(up6)
        conv6 = LeakyReLU()(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Dropout(0.2)(conv6)
        conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
        conv6 = LeakyReLU()(conv6)
        conv6 = BatchNormalization()(conv6)
        conc6 = concatenate([up6, conv6], axis=3)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), padding='same')(up7)
        conv7 = LeakyReLU()(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Dropout(0.2)(conv7)
        conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
        conv7 = LeakyReLU()(conv7)
        conv7 = BatchNormalization()(conv7)
        conc7 = concatenate([up7, conv7], axis=3)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), padding='same')(up8)
        conv8 = LeakyReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Dropout(0.2)(conv8)
        conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
        conv8 = LeakyReLU()(conv8)
        conv8 = BatchNormalization()(conv8)
        conc8 = concatenate([up8, conv8], axis=3)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), padding='same')(up9)
        conv9 = LeakyReLU()(conv9)
        conv9 = Dropout(0.2)(conv9)
        conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
        conv9 = LeakyReLU()(conv9)
        conc9 = concatenate([up9, conv9], axis=3)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc9)

        self.generator = Model(inputs=inputs, outputs=[conv10])


    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.image_height, self.image_width, self.input_channels], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.image_height, self.image_width, self.output_channels], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        self.discriminator = tf.keras.Model(inputs=[inp, tar], outputs=last)


    def compile(self, discriminator_optimizer, generator_optimizer, discriminator_loss, generator_loss, metrics=[]):
        super().compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss


    def train_step(self, data):
        input_image, target = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return {"d_loss": disc_loss, "g_total": gen_total_loss, 'g_gan': gen_gan_loss, 'g_l1': gen_l1_loss, 'dice': dice(target, gen_output)}


    def predict(self, data, batch_size):
        return self.generator.predict(data, batch_size=batch_size)


    def test_step(self, data):
        input_image, target = data
        pred = self.generator(input_image, training=False)
        print(target)
        return {'dice': dice(target, pred)}


    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None):
        self.generator.save(filepath, overwrite, include_optimizer, save_format, signatures, options)


    # def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
    #     self.generator.save_weights(filepath, overwrite, save_format, options)
    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.generator.save_weights(filepath, overwrite, save_format)


    def load_weights(self, filepath, by_name=False, skip_mismatch=False):
        self.generator.load_weights(filepath, by_name, skip_mismatch)


    # def save_model(self,
    #     model, filepath, overwrite=True, include_optimizer=True, save_format=None,
    #     signatures=None, options=None, save_traces=True
    #     ):
    #     self.generator.save_model(
    #         model, filepath, overwrite, include_optimizer, save_format,
    #         signatures, options, save_traces
    #     )
