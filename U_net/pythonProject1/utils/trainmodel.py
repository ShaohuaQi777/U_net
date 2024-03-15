from prepare_data import load_and_preprocess_data
from model import unet_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt


def train_model():
    prepared_data, label_one_hot = load_and_preprocess_data()
    T1_train, T1_test, label_train, label_test = train_test_split(prepared_data, label_one_hot, test_size=0.2,
                                                                  random_state=42)

    model = unet_model()
    model.summary()

    # 数据增强
    data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')
    image_datagen = ImageDataGenerator(**data_gen_args)

    print("Starting training...")
    results = model.fit(image_datagen.flow(T1_train, label_train, batch_size=32),
                        validation_data=(T1_test, label_test),
                        steps_per_epoch=len(T1_train) // 5,
                        epochs=200)

    # 评估模型
    _, accuracy = model.evaluate(T1_test, label_test)
    print("Test accuracy:", accuracy)

    # 模型预测
    # 假设 T1_test 是你的测试数据
    predicted_labels = model.predict(T1_test)

    # 转换概率为类别标签
    predicted_classes = np.argmax(predicted_labels, axis=-1)

    # 选择一个样本进行可视化
    index_to_visualize = 0  # 例如，选择第一个样本

    # 可视化原始图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(T1_test[index_to_visualize].squeeze(), cmap='gray')  # 假设 T1_test 是归一化后的图像数据
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    # 可视化真实标签（如果可用）
    # 假设你有一个相应的真实标签数组 label_test
    plt.subplot(1, 3, 2)
    true_label = np.argmax(label_test[index_to_visualize], axis=-1)  # 只有当label_test是one-hot编码时才需要
    plt.imshow(true_label, cmap='jet', interpolation='nearest')
    plt.title('True Label')
    plt.xticks([])
    plt.yticks([])

    # 可视化预测标签
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_classes[index_to_visualize], cmap='jet', interpolation='nearest')
    plt.title('Predicted Label')
    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == "__main__":
    train_model()
