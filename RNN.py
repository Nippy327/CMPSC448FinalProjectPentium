from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Parameters
img_width, img_height = 64, 64
num_classes = 3

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'CMPSC448FinalProjectPentium-main/data',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

# RNN Model
model = Sequential([
    Reshape((img_width, img_height * 3), input_shape=(img_width, img_height, 3)),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(256)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=20, steps_per_epoch=20)

# Save the model
model.save('rnn_image_classifier.keras')
