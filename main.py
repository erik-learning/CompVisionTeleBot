from telegram.ext import CommandHandler, MessageHandler, Filters, Updater
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf

API_KEY = '*************************************'

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images/255.0, test_images/255.0

label_names = ['Aircraft', 'Automobile', 'Avian', 'Feline', 'Elk', 'Canine', 'Toad', 'Equine', 'Boat', 'Lorry']

neural_net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

def greet_user(update, context):
    update.message.reply_text("Hello there!")

def provide_help(update, context):
    update.message.reply_text("""
    /begin - Begin interaction
    /assist - This guidance
    /initiate_training - Start model training
    """)

def initiate_training(update, context):
    update.message.reply_text("Commencing model training...")
    neural_net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    neural_net.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    neural_net.save('BotModel_Telegram.model')
    update.message.reply_text("Training complete. Please upload a picture.")

def process_text(update, context):
    update.message.reply_text("For better interaction, first train the model and then upload an image.")

def analyze_image(update, context):
    img_file = context.bot.get_file(update.message.photo[-1].file_id)
    img_stream = BytesIO(img_file.download_as_bytearray())
    img_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    
    analyzed_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    analyzed_img = cv2.cvtColor(analyzed_img, cv2.COLOR_RGB2BGR)
    analyzed_img = cv2.resize(analyzed_img, (32,32), interpolation=cv2.INTER_AREA)

    predicted_value = neural_net.predict(np.array([analyzed_img / 255.0]))
    update.message.reply_text(f"I believe this is a {label_names[np.argmax(predicted_value)]}.")

telegram_bot = Updater(API_KEY, use_context=True)
dispatcher = telegram_bot.dispatcher

dispatcher.add_handler(CommandHandler("begin", greet_user))
dispatcher.add_handler(CommandHandler("assist", provide_help))
dispatcher.add_handler(CommandHandler("initiate_training", initiate_training))
dispatcher.add_handler(MessageHandler(Filters.text, process_text))
dispatcher.add_handler(MessageHandler(Filters.photo, analyze_image))

telegram_bot.start_polling()
telegram_bot.idle()
