import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

from losses import dice


class Pix2Pix(tf.keras.Model):
    def __init__(self,image_width,image_height,input_channels,output_channels):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.build_generator()
        self.build_discriminator()

    def downsample(self,filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self,filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
                result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=[self.image_height,self.image_width,self.input_channels])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            self.downsample(128, 4), # (bs, 64, 64, 128)
            self.downsample(256, 4), # (bs, 32, 32, 256)
            self.downsample(512, 4), # (bs, 16, 16, 512)
            self.downsample(512, 4), # (bs, 8, 8, 512)
            self.downsample(512, 4), # (bs, 4, 4, 512)
            self.downsample(512, 4), # (bs, 2, 2, 512)
            self.downsample(512, 4), # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            self.upsample(512, 4), # (bs, 16, 16, 1024)
            self.upsample(256, 4), # (bs, 32, 32, 512)
            self.upsample(128, 4), # (bs, 64, 64, 256)
            self.upsample(64, 4), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channels, 4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='sigmoid') # (bs, 256, 256, 3)

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

        self.generator = tf.keras.Model(inputs=inputs, outputs=x)

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.image_height,self.image_width, self.input_channels], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.image_height,self.image_width,self.output_channels], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x) # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1) # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2) # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

        self.discriminator = tf.keras.Model(inputs=[inp, tar], outputs=last)


    def compile(self,discriminator_optimizer, generator_optimizer,discriminator_loss,generator_loss,metrics=[]):
        super().compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss

    def train_step(self,data):
        input_image, target = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_variables))

        return {"d_loss":disc_loss,"g_total":gen_total_loss,'g_gan':gen_gan_loss,'g_l1':gen_l1_loss,'dice':dice(target,gen_output) }

    def predict(self,data,batch_size):
        return self.generator.predict(data,batch_size=batch_size)

    def test_step(self,data):
        print("TESTE",len(data))
        input_image, target = data
        pred = self.generator(input_image,training=False)
        print(target)
        return {'dice':dice(target,pred) }
    
    def save(self,filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None):
        self.generator.save(filepath, overwrite, include_optimizer, save_format, signatures, options)
    
    def save_weights(self,filepath, overwrite=True, save_format=None) :
        self.generator.save_weights(filepath, overwrite, save_format) 
    
    def load_weights(self,filepath, by_name=False, skip_mismatch=False):
        self.generator.load_weights(filepath, by_name, skip_mismatch)
    
#     def save_model(self,
#         model, filepath, overwrite=True, include_optimizer=True, save_format=None,
#         signatures=None, options=None, save_traces=True
#         ):
#         self.generator.save_model(
#             model, filepath, overwrite, include_optimizer, save_format,
#             signatures, options, save_traces
#         )