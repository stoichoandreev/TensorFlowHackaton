package com.gumtree.tensorflowexample.tensorflow

import android.annotation.TargetApi
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Trace
import android.support.annotation.NonNull
import android.util.Log
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.*

class TensorFlowImageClassifier : Classifier {

    private var inputName: String? = null
    private var outputName: String? = null
    private var inputSize: Int = 0
    private var imageMean: Int = 0
    private var imageStd: Float = 0.0f

    //Buffers.
    private val labels: Vector<String> by lazy { Vector<String>() }
    private var intValues: IntArray? = null
    private var floatValues: FloatArray? = null
    private var outputs: FloatArray? = null
    private var outputNames: Array<String>? = null

    private var inferenceInterface: TensorFlowInferenceInterface? = null

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize     The input size. A square image of inputSize x inputSize is assumed.
     * @param imageMean     The assumed mean of the image values.
     * @param imageStd      The assumed std of the image values.
     * @param inputName     The label of the image input node.
     * @param outputName    The label of the output node.
     * @throws IOException
     */
    companion object {
        private val TAG = "GTImageClassifier"
        // Only return this many results with at least this confidence.
        private val MAX_RESULTS = 3
        private val THRESHOLD = 0.1f

        @Throws(IOException::class)
        fun create(
                @NonNull assetManager: AssetManager,
                @NonNull modelFilename: String,
                @NonNull labelFilename: String,
                inputSize: Int,
                imageMean: Int,
                imageStd: Float,
                inputName: String,
                outputName: String): Classifier {
            val tensorFlowImageClassifier = TensorFlowImageClassifier()
            tensorFlowImageClassifier.inputName = inputName
            tensorFlowImageClassifier.outputName = outputName

            // Read the label names into memory.
            // TODO this reading can be optimized.
            val actualFilename = labelFilename.split("file:///android_asset/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()[1]
            Log.i(TAG, "Reading labels: " + actualFilename)
            val br: BufferedReader? = BufferedReader(InputStreamReader(assetManager.open(actualFilename)))
            br?.readLines()?.forEach {
                tensorFlowImageClassifier.labels.add(it)
            }
            br?.close()

            tensorFlowImageClassifier.inferenceInterface = TensorFlowInferenceInterface()
            if (tensorFlowImageClassifier.inferenceInterface!!.initializeTensorFlow(assetManager, modelFilename) != 0) {
                throw RuntimeException("TF initialization failed")
            }
            // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
            val numClasses = tensorFlowImageClassifier.inferenceInterface!!.graph().operation(outputName).output(0).shape().size(1).toInt()
            Log.i(TAG, "Read " + tensorFlowImageClassifier.labels.size + " labels, output layer size is " + numClasses)

            // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
            // the placeholder node for input in the graphdef typically used does not specify a shape, so it
            // must be passed in as a parameter.
            tensorFlowImageClassifier.inputSize = inputSize
            tensorFlowImageClassifier.imageMean = imageMean
            tensorFlowImageClassifier.imageStd = imageStd

            // Pre-allocate buffers.
            tensorFlowImageClassifier.outputNames = arrayOf(outputName)
            tensorFlowImageClassifier.intValues = IntArray(inputSize * inputSize)
            tensorFlowImageClassifier.floatValues = FloatArray(inputSize * inputSize * 3)
            tensorFlowImageClassifier.outputs = FloatArray(numClasses)

            return tensorFlowImageClassifier
        }
    }

    @TargetApi(18)
    override fun recognizeImage(bitmap: Bitmap): List<Classifier.Recognition> {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage")

        Trace.beginSection("preprocessBitmap")
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (i in intValues!!.indices) {
            val value = intValues!![i]
            floatValues?.set(i * 3 + 0, ((value shr 16 and 0xFF) - imageMean) / imageStd)
            floatValues?.set(i * 3 + 1, ((value shr 8 and 0xFF) - imageMean) / imageStd)
            floatValues?.set(i * 3 + 2, ((value and 0xFF) - imageMean) / imageStd)
        }
        Trace.endSection()

        // Copy the input data into TensorFlow.
        Trace.beginSection("fillNodeFloat")
        inferenceInterface!!.fillNodeFloat(
                inputName!!, intArrayOf(1, inputSize, inputSize, 3), floatValues)
        Trace.endSection()

        // Run the inference call.
        Trace.beginSection("runInference")
        inferenceInterface!!.runInference(outputNames!!)
        Trace.endSection()

        // Copy the output Tensor back into the output array.
        Trace.beginSection("readNodeFloat")
        inferenceInterface!!.readNodeFloat(outputName, outputs)
        Trace.endSection()

        // Find the best classifications.
        val priorityQueue = PriorityQueue(3,
                Comparator<Classifier.Recognition> { lhs, rhs ->
                    //                     Intentionally reversed to put high confidence at the head of the queue.
                    java.lang.Float.compare(rhs.confidence!!, lhs.confidence!!)
                })

        outputs!!.indices
                .filter { outputs!![it] > THRESHOLD }
                .mapTo(priorityQueue) {
                    Classifier.Recognition(
                            "" + it, if (labels.size > it) labels[it] else "unknown", outputs!![it], null)
                }

        val recognitions = ArrayList<Classifier.Recognition>()
        val recognitionsSize = Math.min(priorityQueue.size, MAX_RESULTS)
        for (i in 0 until recognitionsSize) {
            recognitions.add(priorityQueue.poll())
        }
        Trace.endSection() //"recognizeImage"
        return recognitions
    }

    override fun enableStatLogging(debug: Boolean) {
        inferenceInterface?.enableStatLogging(debug)
    }

    override fun getStatString(): String? =
            inferenceInterface?.statString

    override fun close() {
        inferenceInterface!!.close()
    }
}