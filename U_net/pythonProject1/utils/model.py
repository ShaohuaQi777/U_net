from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

def unet_model(input_size=(256, 256, 1), num_classes=6):
    inputs = Input(input_size)
    # 编码器
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    p5 = MaxPooling2D((2, 2))(c5)

    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    p6 = MaxPooling2D((2, 2))(c6)

    c7 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p6)
    p7 = MaxPooling2D((2, 2))(c7)

    # 底部
    c8 = Conv2D(2048, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p7)

    # 解码器
    u9 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c7])
    c9 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)

    u10 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c6])
    c10 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)

    u11 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c5])
    c11 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)

    u12 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c11)
    u12 = concatenate([u12, c4])
    c12 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u12)

    u13 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c12)
    u13 = concatenate([u13, c3])
    c13 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u13)

    u14 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c13)
    u14 = concatenate([u14, c2])
    c14 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u14)

    u15 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c14)
    u15 = concatenate([u15, c1], axis=3)
    c15 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u15)

    # 输出层
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c15)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
