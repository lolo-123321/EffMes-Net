# -*- coding:utf-8 -*-
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Concatenate, LeakyReLU, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception

IMGWIDTH = 256

class Classifier:
    def __init__(self):
        self.model = None

    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class MesoXceptionNet(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Proper loss for binary classification
            metrics=['accuracy']
        )

    def init_model(self):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))

        # Meso4 branch
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        meso_features = Flatten()(x4)
        meso_features = Dropout(0.5)(meso_features)
        meso_features = Dense(16)(meso_features)
        meso_features = LeakyReLU(alpha=0.1)(meso_features)

        # Xception branch
        xception_net = Xception(include_top=False, input_shape=(IMGWIDTH, IMGWIDTH, 3), weights='imagenet')

        # Partially unfreeze top Xception layers
        for layer in xception_net.layers[:-30]:
            layer.trainable = False
        for layer in xception_net.layers[-30:]:
            layer.trainable = True

        xception_features = GlobalAveragePooling2D()(xception_net(x))

        # Concatenate features
        combined_features = Concatenate()([meso_features, xception_features])
        combined_features = Dropout(0.5)(combined_features)

        # Output layer
        output = Dense(1, activation='sigmoid')(combined_features)

        return KerasModel(inputs=x, outputs=output)
