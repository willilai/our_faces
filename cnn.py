import tensorflow.keras.utils as utils
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

train = utils.image_dataset_from_directory(
    'images',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (501, 501),
    seed = 420,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'images',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (501, 501),
    seed = 420,
    validation_split = 0.3,
    subset = 'validation',
)

class Net():
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(layers.ZeroPadding2D(
            padding = ((4,4), (4, 4)),
            input_shape = input_shape,
        ))
        self.model.add(layers.Conv2D(
            8,            # filters
            25,           # kernels
            strides = 11, # step size
            activation = 'relu', 
        ))# output: 45 X 45 X 8
        self.model.add(layers.ZeroPadding2D(
            padding = ((0,1), (0, 1)),
            input_shape = input_shape,
        ))# output: 46 X 46 X 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2
        ))# output: 23 X 23 X 8
        self.model.add(layers.Conv2D(
            8,            # filters
            3,           # kernels
            strides = 1, # step size
            activation = 'relu', 
        ))# output: 21 X 21 X 8
        self.model.add(layers.ZeroPadding2D(
            padding = ((0,1), (0, 1)),
            input_shape = input_shape,
        ))# output: 22 X 22 X 8
        self.model.add(layers.MaxPool2D(
            pool_size = 2
        ))# output: 11 X 11 X 8
        self.model.add(layers.Flatten(
        ))# output: 968
        self.model.add(layers.Dense(
            1024, 
            activation = 'relu', 
        ))
        self.model.add(layers.Dense(
            256,
            activation = 'relu', 
        ))
        self.model.add(layers.Dense(
            64, 
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            17,                     # Exactly equal to number of classes (ask Dr.J about this number)
            activation = 'softmax', # Always use softmax on your last layer
        ))
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = 'accuracy',
        )
    def __str__(self):
        self.model.summary()
        return ""

net = Net((501,501,3))
print(net)

net.model.fit(
    train,
    batch_size = 32,
    epochs = 40,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)

# net.model.save("faces_model_save")