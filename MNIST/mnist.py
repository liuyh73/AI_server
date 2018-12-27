import os
import struct
import random
import numpy as np
import matplotlib.pyplot as plt

image_dimen = 28
dimens=[784, 30, 10]
iteration = 3

def load_mnist(path, kind = "train"):
  labels_path = os.path.join(path, "%s-labels.idx1-ubyte" % kind)
  images_path = os.path.join(path, "%s-images.idx3-ubyte" % kind)

  with open(labels_path, 'rb') as lpath:
    magic, n = struct.unpack('>II', lpath.read(8))
    labels = np.fromfile(lpath, dtype=np.uint8)
  
  with open(images_path, 'rb') as ipath:
    magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
    images = np.fromfile(ipath, dtype=np.uint8).reshape(len(labels), image_dimen*image_dimen)

  return images, labels

def draw_img(img):
  fig, ax = plt.subplots(
    sharex = True,
    sharey = True,
  )
  ax.imshow(img.reshape(28,28), cmap="Greys", interpolation="nearest")
  ax.set_xticks([])
  ax.set_yticks([])
  plt.tight_layout()
  plt.show()

def get_random(dimen1, dimen2):
  l = []
  for i in range(dimen1):
    l.append([])
    for j in range(dimen2):
      l[i].append(random.uniform(-2.4/(image_dimen*image_dimen), 2.4/(image_dimen*image_dimen)))
  return l

def init_matrix():
  return np.array(get_random(dimens[0], dimens[1])), np.array(get_random(dimens[1], dimens[2]))

def sigmoid(nodes):
  return 1.0/(1.0+np.exp(-1 * nodes))

def expected(label):
  li = [0.01 for i in range(10)]
  li[label] = 0.99
  return np.array(li)

def train(inputToHidden, hiddenToOutput):
  images, labels = load_mnist("data", "train")
  count, length = 0, int(images.size/(image_dimen*image_dimen))
  for i in range(length):
    for k in range(iteration):
      # 1*784
      input_layer_excited = sigmoid(np.array([images[i]]))
      # 1*30
      hidden_layout_excited = sigmoid(np.dot(input_layer_excited, inputToHidden))
      # 1*10
      output_layout_excited = sigmoid(np.dot(hidden_layout_excited, hiddenToOutput))
      expectedValue = expected(labels[i])
      # 1*10
      deviation = expectedValue - output_layout_excited
      # 1*10
      deviationOutputFix = output_layout_excited * (1-output_layout_excited) * deviation
      # 1*30
      deviationHiddenFix = hidden_layout_excited * (1-hidden_layout_excited) * np.dot(deviationOutputFix, hiddenToOutput.T)
      # 30*10
      hiddenToOutput = hiddenToOutput + 0.05 * np.dot(hidden_layout_excited.T, deviationOutputFix)
      # 784*30
      inputToHidden = inputToHidden + 0.05 * np.dot(input_layer_excited.T, deviationHiddenFix)
    count += 1
    if count%1000 == 0:
      print("已训练",count,"张图片")
  return inputToHidden, hiddenToOutput

def test(inputToHidden, hiddenToOutput):
  images, labels = load_mnist("data", "t10k")
  count, length = 0, int(images.size/image_dimen/image_dimen)
  for i in range(length):
    # 1*784
    input_layer_excited = sigmoid(np.array([images[i]]))
    # 1*30
    hidden_layout_excited = sigmoid(np.dot(input_layer_excited, inputToHidden))
    # 1*10
    output_layout_excited = sigmoid(np.dot(hidden_layout_excited, hiddenToOutput))
    if np.argmax(output_layout_excited[0]) == labels[i]:
      count+=1
  print(count/length)

def main():
  inputToHidden, hiddenToOutput = init_matrix()
  inputToHidden, hiddenToOutput = train(inputToHidden, hiddenToOutput)
  test(inputToHidden, hiddenToOutput)

if __name__ == '__main__':
  main()