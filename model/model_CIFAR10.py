import tensorflow as tf
def create_model_CIFAR10():
    Resnet = tf.keras.models.load_model("model/h5/ResNET50.h5")
    """
    for layer in Resnet.layers:
        layer.trainable = False
    """
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.UpSampling2D((7, 7))(inputs)
    x = Resnet(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation="relu")(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    output = tf.keras.layers.Dense(units=10, activation="softmax")(x)

    model_CIFAR10 = tf.keras.Model(inputs=inputs, outputs=output)
    return model_CIFAR10,tf.keras.losses.SparseCategoricalCrossentropy(),tf.keras.optimizers.Adam(),tf.keras.metrics.SparseCategoricalAccuracy()

"""
def create_model_CIFAR10():

  
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, (3,3), activation = "relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation="relu")(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    output = tf.keras.layers.Dense(units=10, activation="softmax")(x)
    
    model_CIFAR10 = tf.keras.Model(inputs=inputs, outputs=output)
    return model_CIFAR10,tf.keras.losses.SparseCategoricalCrossentropy(),tf.keras.optimizers.Adam(),tf.keras.metrics.SparseCategoricalAccuracy()
 

if __name__ == "__main__":
  model, _ , _ , _ , = create_model_CIFAR10()

"""
