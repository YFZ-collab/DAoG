import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras import layers, optimizers
from tensorflow.keras import backend as K


# Download the preprocessor model for BERT (en-uncased) from the following Kaggle link:
# https://www.kaggle.com/models/tensorflow/bert/tensorFlow2/en-uncased-preprocess
preprocessor_path = "D:/models/tfbert"

# Download the BERT encoder model (en-uncased, 12-layer, 768 hidden units, 12 attention heads) from the following Kaggle link:
# https://www.kaggle.com/models/tensorflow/bert/tensorFlow2/en-uncased-l-12-h-768-a-12/4?tfhub-redirect=true
encoder_path = "D:/models/bert_en_uncased"


preprocessor = hub.KerasLayer(preprocessor_path, name="preprocessing")
bert_encoder = hub.KerasLayer(encoder_path, trainable=False, name="BERT_encoder")


def load_sst2_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=0)
    df = df[["sentence", "label"]]
    df = df[pd.to_numeric(df["label"], errors='coerce').notna()]
    return df


# Download the SST-2 dataset from the following link:
# https://dl.fbaipublicfiles.com/glue/data/SST-2.zip

# Load the training dataset
train_data = load_sst2_data("D:/DAoG/GLUE/datasets/SST-2/train.tsv")

# Load the validation dataset
test_data = load_sst2_data("D:/DAoG/GLUE/datasets/SST-2/dev.tsv")

sampled_train_data = train_data.sample(frac=0.1, random_state=42)

train_sentences = sampled_train_data["sentence"].astype(str)
train_labels = pd.to_numeric(sampled_train_data["label"]).astype(np.float32)

val_sentences = test_data["sentence"].astype(str)
val_labels = pd.to_numeric(test_data["label"]).astype(np.float32)




def build_model(optimizer):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    encoder_inputs = preprocessor(text_input)
    outputs = bert_encoder(encoder_inputs)

    sequence_output = outputs["sequence_output"]

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.0005))(sequence_output)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    final_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=text_input, outputs=final_output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


class DAoG(optimizers.Optimizer):
  def __init__(self,
                 learning_rate=1.0,
                 name="DAoG",
                 **kwargs
  ):
    super(DAoG, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self.learning_rate = learning_rate
  def _create_slots(self, var_list):
      for var in var_list:
        self.add_slot(var, 'm', initializer=tf.constant(0.0))
        self.add_slot(var, 'wm', initializer=tf.constant(0.0))
        self.add_slot(var, 'ini')

  def _resource_apply_dense(self, grad, var):
      var_dtype = var.dtype.base_dtype
      local_step = K.cast(self.iterations + 1, var_dtype)
      ini = self.get_slot(var, 'ini')

      m = self.get_slot(var, 'm')
      wm = self.get_slot(var, 'wm')

      condition = tf.equal(local_step, 1)
      tf.cond(
          condition,
          lambda: K.update(ini, var),
          lambda: K.update(ini, ini)
      )

      tf.cond(
          condition,
          lambda: K.update(m, tf.norm(ini)),
          lambda: K.update(m, tf.maximum(m, tf.norm(var - ini))),
      )

      norm_x = m

      norm_grad = K.update(wm, wm +tf.reduce_sum(tf.square(grad)))

      et = tf.pow(norm_x, 1 / (4 )) / tf.pow(norm_grad, 1 / (5 - 0.06 * local_step / 106))

      var_update = var - self.learning_rate * et * grad

      return K.update(var, var_update)

  def _resource_apply_sparse(self, grad, var, indices):
          grad = tf.IndexedSlices(grad, indices, K.shape(var))
          grad = tf.convert_to_tensor(grad)
          return self._resource_apply_dense(grad, var)

  def get_config(self):
      config = {'learning_rate': self._serialize_hyperparameter('learning_rate'),
                }
      base_config = super(DAoG, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

class DoWG(optimizers.Optimizer):
  def __init__(self,
                 learning_rate=1.0,
                 name="DoWG",
                 **kwargs
  ):
    super(DoWG, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self.learning_rate = learning_rate
  def _create_slots(self, var_list):
      for var in var_list:
        self.add_slot(var, 'm', initializer=tf.constant(0.0))
        self.add_slot(var, 'wm', initializer=tf.constant(0.0))
        self.add_slot(var, 'ini')

  def _resource_apply_dense(self, grad, var):
      var_dtype = var.dtype.base_dtype
      local_step = K.cast(self.iterations + 1, var_dtype)
      ini = self.get_slot(var, 'ini')

      m = self.get_slot(var, 'm')
      wm = self.get_slot(var, 'wm')

      condition = tf.equal(local_step, 1)
      tf.cond(
          condition,
          lambda: K.update(ini, var),
          lambda: K.update(ini, ini)
      )

      tf.cond(
          condition,
          lambda: K.update(m, 1e-4*(1+tf.norm(ini))),
          lambda: K.update(m, tf.maximum(m, tf.norm(var - ini))),
      )

      norm_x = m

      norm_grad = K.update(wm, wm +tf.square(norm_x)*tf.reduce_sum(tf.square(grad)))

      et = tf.square(norm_x) / tf.pow(norm_grad, 1 / 2)
      var_update = var - self.learning_rate * et * grad

      return K.update(var, var_update)

  def _resource_apply_sparse(self, grad, var, indices):
          grad = tf.IndexedSlices(grad, indices, K.shape(var))
          grad = tf.convert_to_tensor(grad)
          return self._resource_apply_dense(grad, var)

  def get_config(self):
      config = {'learning_rate': self._serialize_hyperparameter('learning_rate'),
                }
      base_config = super(DoWG, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

class DoG(optimizers.Optimizer):
  def __init__(self,
                 learning_rate=1.0,
                 name="DoG",
                 **kwargs
  ):
    super(DoG, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self.learning_rate = learning_rate
  def _create_slots(self, var_list):
      for var in var_list:
        self.add_slot(var, 'm', initializer=tf.constant(0.0))
        self.add_slot(var, 'wm', initializer=tf.constant(0.0))
        self.add_slot(var, 'ini')

  def _resource_apply_dense(self, grad, var):
      var_dtype = var.dtype.base_dtype
      local_step = K.cast(self.iterations + 1, var_dtype)
      ini = self.get_slot(var, 'ini')

      m = self.get_slot(var, 'm')
      wm = self.get_slot(var, 'wm')

      condition = tf.equal(local_step, 1)
      tf.cond(
          condition,
          lambda: K.update(ini, var),
          lambda: K.update(ini, ini)
      )

      tf.cond(
          condition,
          #lambda: K.update(m, 1e-6*(tf.norm(ini)+1)),
          lambda: K.update(m, 1e-1*(tf.norm(ini)+1)),
          lambda: K.update(m, tf.maximum(m, tf.norm(var - ini))),
      )

      norm_x = m

      norm_grad = K.update(wm, wm +tf.reduce_sum(tf.square(grad)))

      et = tf.pow(norm_x, 1) / tf.pow(norm_grad, 1 /2)

      var_update = var - self.learning_rate * et * grad

      return K.update(var, var_update)

  def _resource_apply_sparse(self, grad, var, indices):
    grad = tf.IndexedSlices(grad, indices, K.shape(var))
    grad = tf.convert_to_tensor(grad)
    return self._resource_apply_dense(grad, var)

  def get_config(self):
      config = {'learning_rate': self._serialize_hyperparameter('learning_rate'),
                }
      base_config = super(DoG, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

class Adam(optimizers.Optimizer):
  def __init__(
          self,
          learning_rate=0.001,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-8,
          bias_correction=True,
          name="Adam",
          **kwargs
  ):
    super(Adam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self.learning_rate = learning_rate
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or K.epislon()
    self.bias_correction = bias_correction

    if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        self.lr_schedule = learning_rate
    else:
        self.lr_schedule = None

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'm')
      self.add_slot(var, 'wm')

  def _resource_apply_dense(self, grad, var):

    actual_lr = self.lr_schedule(self.iterations) if self.lr_schedule else self.learning_rate
    var_dtype = var.dtype.base_dtype
    m = self.get_slot(var, 'm')
    wm = self.get_slot(var, 'wm')
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    local_step = K.cast(self.iterations + 1, var_dtype)
    beta_1_t_power = K.pow(beta_1_t, local_step)
    beta_2_t_power = K.pow(beta_2_t, local_step)

    m = K.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
    wm = K.update(wm, beta_2_t * wm + (1 - beta_2_t) * K.square(grad))


    m = m / (1.0 - beta_1_t_power)
    wm = wm / (1.0 - beta_2_t_power)

    et = actual_lr / (K.sqrt(wm) + self.epsilon)
    var_t = var - et*m

    return K.update(var, var_t)

  def _resource_apply_sparse(self, grad, var, indices):
    grad = tf.IndexedSlices(grad, indices, K.shape(var))
    grad = tf.convert_to_tensor(grad)
    return self._resource_apply_dense(grad, var)

  def get_config(self):
    config = {
      'learning_rate': self._serialize_hyperparameter('learning_rate'),
      'beta_1': self._serialize_hyperparameter('beta_1'),
      'beta_2': self._serialize_hyperparameter('beta_2'),
      'epsilon': self.epsilon,
      'bias_correction': self.bias_correction,
    }
    base_config = super(Adam, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class LearningRateEpochLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_lr_history = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_lr_history[epoch] = []
        self.current_epoch = epoch
    def on_train_batch_end(self, batch, logs=None):
        batch_lrs = []

        var = self.model.trainable_variables[0]
        m = self.model.optimizer.get_slot(var, "m")
        wm = self.model.optimizer.get_slot(var, "wm")
        if m is not None and wm is not None:
            m_val = m.numpy().flatten()[0]
            wm_val = wm.numpy().flatten()[0]
            lr = self.model.optimizer._get_hyper("learning_rate")
            local_step = self.model.optimizer.iterations.numpy()


            # if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            #     self.lr_schedule = lr
            # else:
            #     self.lr_schedule = None
            # actual_lr = self.lr_schedule(self.model.optimizer.iterations) if self.lr_schedule else self.learning_rate


            if hasattr(lr, "numpy"):
                lr = lr.numpy()
            effective_lr = lr * np.power(m_val, 1/4 ) / np.power(wm_val, 1 / (5- 0.06 * local_step / 106))  #DAoG

            #effective_lr = lr * np.square(m_val) / np.power(wm_val, 1 / 2)       #DoWG

            #effective_lr = lr * m_val / np.power(wm_val, 1 / 2)  # DoG

            #effective_lr = actual_lr.numpy() / (np.sqrt(wm_val) + 1e-10)  # Adam

            batch_lrs.append(np.mean(effective_lr))

        if batch_lrs:
            self.epoch_lr_history[self.current_epoch].append(np.mean(batch_lrs))

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_lr_history:
            epoch_effective_lr = np.mean(self.epoch_lr_history[epoch]) if self.epoch_lr_history[epoch] else None
        else:
            epoch_effective_lr = None

        self.epoch_lr_history[self.current_epoch].append(epoch_effective_lr)
        print(f"Epoch {epoch} effective lr: {epoch_effective_lr}")
    def on_train_end(self, logs=None):
        with open("learning_rate_history.json", "w") as f:
            json.dump(self.epoch_lr_history, f, indent=4)

        avg_lrs = [np.mean(v) for v in self.epoch_lr_history.values() if v]
        epochs = list(self.epoch_lr_history.keys())

        plt.plot(epochs, avg_lrs, marker='o')
        plt.title("First Element Learning Rate Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Effective Learning Rate (First Element)")
        plt.show()



lr_epoch_logger = LearningRateEpochLogger()


lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=5300,
    alpha=0.01
)

optimizers = {
    'DAoG': DAoG(learning_rate=1.0),
    #'DoWG':DoWG(learning_rate=1.0),
    #'DoG': DoG(learning_rate=1.0)
    #'Adam': Adam(learning_rate=lr_schedule),
    #'SGD': tf.keras.optimizers.SGD(learning_rate=lr_schedule),
  }


history = {name: {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []} for name in
             optimizers.keys()}


for opt_name, optimizer in optimizers.items():
    print(f"\nTraining with {opt_name} optimizer...")
    model = build_model(optimizer)

    hist = model.fit(
        train_sentences, train_labels,
        validation_data=(val_sentences, val_labels),
        epochs=50, batch_size=64,
        callbacks=[lr_epoch_logger],
        verbose=1
    )

    history[opt_name]['train_loss'] = hist.history['loss']
    history[opt_name]['train_accuracy'] = [acc * 100 for acc in hist.history['accuracy']]
    history[opt_name]['val_loss'] = hist.history['val_loss']
    history[opt_name]['val_accuracy'] = [acc * 100 for acc in hist.history['val_accuracy']]

for opt_name, results in history.items():
    file_name = f"{opt_name.replace(' ', '_').replace('(', '').replace(')', '')}_history.json"
    with open(file_name, 'w') as f:
        json.dump(results, f)

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
for opt_name in optimizers.keys():
    plt.plot(history[opt_name]['train_loss'], label=f"{opt_name} ")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
for opt_name in optimizers.keys():
    plt.plot(history[opt_name]['train_accuracy'], label=f"{opt_name} ")
plt.xlabel("Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.title("Training Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
for opt_name in optimizers.keys():
    plt.plot(history[opt_name]['val_loss'], label=f"{opt_name} ")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Testing Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
for opt_name in optimizers.keys():
    plt.plot(history[opt_name]['val_accuracy'], label=f"{opt_name} ")
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.title("Testing Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
