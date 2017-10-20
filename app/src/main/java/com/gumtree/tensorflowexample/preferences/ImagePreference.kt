package com.gumtree.tensorflowexample.preferences

object ImagePreference {
    val INPUT_SIZE = 224
    val IMAGE_MEAN = 117
    val IMAGE_STD = 1f
    val INPUT_NAME = "input"
    val OUTPUT_NAME = "output"
    val MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb"
    val LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt"
}