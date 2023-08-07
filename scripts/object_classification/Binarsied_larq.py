# Guidance from this link: https://docs.larq.dev/zoo/tutorials/#evaluate-quicknet-with-tensorflow-datasets
import os
import sys

sys.path.append('../../')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # To ensure tensorflow uses CPU

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import larq_zoo as lqz
from urllib.request import urlopen
from PIL import Image

import time
import pyRAPL
import psutil

vram  = psutil.virtual_memory()
initial_ram_usage_gegabytes = (vram.total - vram.available)   / 1024 ** 3

initial_cpu_utilistation_percentage = psutil.cpu_percent(5) # Measruing insitial CPU utilisation over 5 seconds.

initial_machine_status_df= {'ram_usage': initial_ram_usage_gegabytes,
                               'CPU_utilisation': initial_cpu_utilistation_percentage}

initial_machine_status = initial_machine_status_df


def preprocess(data):
    img = lqz.preprocess_input(data["image"])
    label = tf.one_hot(data["label"], 1000)
    return img, label 

# == Prepare dataset
dataset = (
    tfds.load("imagenet2012:5.1.0", split=tfds.Split.VALIDATION)
    .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(128)
    .prefetch(1)
)

# === Prepare single input for measurement of non accuracy sensitive metrics
img_path = "https://raw.githubusercontent.com/larq/zoo/master/tests/fixtures/elephant.jpg"

with urlopen(img_path) as f:
    img = Image.open(f).resize((224, 224))

x = tf.keras.preprocessing.image.img_to_array(img)
x = lqz.preprocess_input(x)
x = np.expand_dims(x, axis=0)

# === Load model
# model = lqz.literature.BinaryResNetE18(weights="imagenet")
# model = lqz.sota.QuickNet(weights="imagenet")
# model = lqz.literature.XNORNet(weights="imagenet")
# model = lqz.literature.RealToBinaryNet(weights="imagenet")
model = lqz.literature.BinaryDenseNet45(weights="imagenet")



# ======================================================================
# === Measure Accuracy
evalute_accuracy_flag = False
if evalute_accuracy_flag == True:
    model.compile(
        optimizer="sgd",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],#, "top_k_categorical_accuracy"],
    )

    model.evaluate(dataset)

# ======================================================================
# === Measure Latency 
num_samples=1000

start_time = time.time()

for _ in range(num_samples):
    _ = model.predict(x)

end_time = time.time()

elapsed_time = end_time - start_time
elapsed_time_ave = elapsed_time / num_samples
print("Latency (ms) = ", elapsed_time_ave*1000)  # output in ms


# ======================================================================
# === Measure CPU Usage
# Measure before utilsiation before inference.
utilisation_before = initial_machine_status['CPU_utilisation']

# Measure utilistion during inference.
num_samples=1000
num_warmups=100

utilisation_during = 0

for _ in range(num_warmups):
    _ = model.predict(x)


psutil.cpu_percent(interval=0, percpu=False) # Running command which should be the reference meaurement for next time the command is called.
for _ in range(num_samples):
    _ = model.predict(x)
    utilisation_during += psutil.cpu_percent(interval=0, percpu=False)
utilisation_during = utilisation_during/num_samples

# Subtract to get utilisation due to inference
cpu_utilisation_by_model = utilisation_during - utilisation_before
print("CPU utilisation (GB) = ",cpu_utilisation_by_model)


# ======================================================================
# === Measure RAM Usage
# Measure before utilsiation before inference.
usage_before = initial_machine_status['ram_usage']

# Measure utilistion during inference.
num_samples=1000
num_warmups=100

usage_during = 0

for _ in range(num_warmups):
    _ = model.predict(x)

for _ in range(num_samples):
    _ = model.predict(x)
    vram  = psutil.virtual_memory()
    usage_during = max(usage_during,((vram.total - vram.available) / 1024 ** 3))

# Subtract to get utilisation due to inference
ram_usage_by_model = usage_during - usage_before

print("RAM usage (GB) = ", ram_usage_by_model) 


# ======================================================================
# === Measure Energy and Power
# Warm up model
num_warmups = 100
for _ in range(num_warmups):
    _ = model.predict(x)

# Runs to measure energy
num_samples = 1000

pyRAPL.setup() 
meter = pyRAPL.Measurement('')
meter.begin()

for _ in range(num_samples):
    _ = model.predict(x)

meter.end()

duration        = meter.result.duration # micro seconds
energy_measured = (meter.result.pkg[0] + meter.result.dram[0]) # micro Joules 
energy_ave      = energy_measured/(1E6 * num_samples) # Joules/predicition 
power_ave       = energy_measured/duration # Watts

print("Energy (Joules/Prediction) = ", energy_ave) 
print("Power (Watts) = ", power_ave) 


# === Measure MAC, Params Count, Disk size, and CHATS are obtained 
#     from data provided by larq:(https://docs.larq.dev/zoo/). 
