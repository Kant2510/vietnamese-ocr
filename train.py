import tensorflow as tf
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric

from model import DefinedModel
from modelConfigs import ModelConfigs
from pathConfigs import *
from tqdm import tqdm

data_path = os.path.join(TRAIN_DATASET_PATH, 'images')
label_path = os.path.join(TRAIN_DATASET_PATH, 'annotations')

dataset, vocab, max_len = [], set(), 0
for filename in tqdm(os.listdir(label_path)):
    # Read each file .txt in folder annotations
    content = open(os.path.join(label_path, filename),
                   'r', encoding='utf-8').readlines()
    for line in content:
        line_split = line.split('\t')
        label = line_split[-1].rstrip('\n')

        # Check path whether exists or not
        check_path = os.path.join(data_path, line_split[0])
        if not os.path.exists(check_path):
            print('File not found:', check_path)
            continue

        dataset.append([check_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

# Storing model configurations
configs = ModelConfigs()
configs.vocab = ''.join(vocab)
configs.max_text_length = max_len
configs.save()

strategy = tf.distribute.MirroredStrategy()

# Create data provider
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size*strategy.num_replicas_in_sync,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[ImageResizer(configs.width, configs.height, keep_aspect_ratio=True), LabelIndexer(
        configs.vocab), LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))],
)
train_data_provider, val_data_provider = data_provider.split(split=0.9)
train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
]
print('Total GPUs:', strategy.num_replicas_in_sync)
with strategy.scope():
    model = DefinedModel(
        input_dim=(configs.height, configs.width, 3),
        output_dim=len(configs.vocab),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=configs.learning_rate),
        loss=CTCloss(),
        metrics=[CERMetric(vocabulary=configs.vocab),
                 WERMetric(vocabulary=configs.vocab)],
        run_eagerly=False,
    )
# model.summary()

# Callbacks
earlystopper = EarlyStopping(
    monitor='val_CER', patience=20, verbose=1, mode='min')
checkpoint = ModelCheckpoint(
    f'{configs.model_path}/model.h5',
    monitor='val_CER',
    verbose=1,
    save_best_only=True,
    mode='min',
)
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f'{configs.model_path}/logs', update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_CER', factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode='auto')
model2onnx = Model2onnx(f'{configs.model_path}/model.h5')

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[
        earlystopper,
        checkpoint,
        trainLogger,
        reduceLROnPlat,
        tb_callback,
        model2onnx,
    ],
    workers=configs.train_workers,
)

# Save train and val data provider
train_data_provider.to_csv(os.path.join(configs.model_path, 'train.csv'))
val_data_provider.to_csv(os.path.join(configs.model_path, 'val.csv'))
