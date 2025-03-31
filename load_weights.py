# load_weights.py
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    TimeDistributed,
    ConvLSTM2D,
    BatchNormalization,
    Conv2D,
    Input,
    Lambda,
    Resizing,
    Conv2DTranspose,
    Concatenate,
    Dropout,
)
import tensorflow as tf  # type: ignore


def build_cloud_model():
    # Input shape (8 timesteps, 333x333 radar frames)
    input_shape = (8, 333, 333, 1)
    inputs = Input(shape=input_shape)

    # --- STEP 1: Preprocess (resize + convert to 3 channels) ---
    x = TimeDistributed(Resizing(352, 352))(inputs)  # Resize for VGG
    x = Lambda(lambda t: tf.repeat(t, repeats=3, axis=-1))(x)  # Grayscale → RGB

    # --- STEP 2: VGG Encoder (TimeDistributed) ---
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=(352, 352, 3))
    vgg.trainable = False

    # Get skip connections from VGG layers
    block3_out = TimeDistributed(Model(vgg.input, vgg.get_layer("block3_pool").output))(
        x
    )  # 44x44
    block4_out = TimeDistributed(Model(vgg.input, vgg.get_layer("block4_pool").output))(
        x
    )  # 22x22

    # Final encoder output
    encoded = TimeDistributed(vgg)(x)  # 11x11

    # --- STEP 3: ConvLSTM ---
    x = ConvLSTM2D(
        filters=512, kernel_size=(3, 3), padding="same", return_sequences=False
    )(encoded)
    x = BatchNormalization()(x)

    # --- STEP 4: Decoder (U-Net style with skip connections) ---
    # Upsample 11x11 → 22x22
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(
        x
    )
    x = Concatenate()([x, block4_out[:, -1]])  # Connect block4_pool
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Upsample 22x22 → 44x44
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(
        x
    )
    x = Concatenate()([x, block3_out[:, -1]])  # Connect block3_pool
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Upsample 44x44 → 88x88
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(
        x
    )
    x = BatchNormalization()(x)

    # Upsample 88x88 → ~333x333 (via resize)
    x = Lambda(lambda t: tf.image.resize(t, size=(333, 333), method="bilinear"))(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)

    # Final output layer
    output = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # --- STEP 5: Compile ---
    model = Model(inputs, output)

    # Combined MSE + SSIM loss
    def combined_loss(y_true, y_pred):
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return 0.5 * mse_loss + 0.5 * ssim_loss

    # --- STEP 5: Compile with Perceptual Loss ---
    # Load VGG16 for perceptual loss (grayscale → RGB conversion inside loss)
    vgg_perceptual = VGG16(
        weights="imagenet", include_top=False, input_shape=(333, 333, 3)
    )
    vgg_perceptual.trainable = False

    # Choose a mid-level layer for perceptual features
    percep_layer = Model(
        vgg_perceptual.input, vgg_perceptual.get_layer("block3_conv3").output
    )

    def perceptual_loss(y_true, y_pred):
        # Convert grayscale to 3 channels inside loss
        y_true_rgb = tf.repeat(y_true, repeats=3, axis=-1)
        y_pred_rgb = tf.repeat(y_pred, repeats=3, axis=-1)

        # Extract perceptual features
        feat_true = percep_layer(y_true_rgb)
        feat_pred = percep_layer(y_pred_rgb)

        # Compute perceptual loss (L2 loss)
        return tf.reduce_mean(tf.square(feat_true - feat_pred))

    # --- Combined Loss ---
    def combined_loss_v2(y_true, y_pred):
        # SSIM Loss
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        # MSE Loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        # Perceptual Loss
        perc_loss = perceptual_loss(y_true, y_pred)

        # Weighted sum
        return 0.1 * mse_loss + 0.1 * ssim_loss + 0.01 * perc_loss

    model.compile(optimizer="adam", loss=combined_loss_v2)

    model.summary()
    return model


def load_cloud_model(weights_path="cloud_model.keras"):
    model = build_cloud_model()
    model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    # ทดสอบการโหลดโมเดล
    model = load_cloud_model()
    model.summary()
