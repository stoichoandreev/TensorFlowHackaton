package com.gumtree.tensorflowexample

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.view.LayoutInflater
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import com.flurgle.camerakit.CameraListener
import kotlinx.android.synthetic.main.activity_main.*
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private val MIN_PRICE = 100
        private val MAX_PRICE = 300
        private val INPUT_SIZE = 224
        private val IMAGE_MEAN = 117
        private val IMAGE_STD = 1f
        private val INPUT_NAME = "input"
        private val OUTPUT_NAME = "output"
        private val MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb"
        private val LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt"
    }

    private var classifier: Classifier? = null
    private val executor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private lateinit var picture: ByteArray
    private val rand: Random by lazy(LazyThreadSafetyMode.NONE) { Random() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        camera_view!!.setCameraListener(object : CameraListener() {
            override fun onPictureTaken(picture: ByteArray) {
                super.onPictureTaken(picture)

                this@MainActivity.picture = picture

                var bitmap = BitmapFactory.decodeByteArray(picture, 0, picture.size)
                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
                image_view_result!!.setImageBitmap(bitmap)
                val results = classifier!!.recognizeImage(bitmap)
                displayImageRecognitionResult(results)
            }
        })

//        button_toggle_camera!!.setOnClickListener { camera_view!!.toggleFacing() }
        button_detect_object!!.setOnClickListener { camera_view!!.captureImage() }

        initTensorFlowAndLoadModel()
    }

    override fun onResume() {
        super.onResume()
        camera_view!!.start()
    }

    override fun onPause() {
        camera_view!!.stop()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.execute({ classifier!!.close() })
    }

    private fun initTensorFlowAndLoadModel() {
        executor.execute({
            try {
                classifier = TensorFlowImageClassifier.create(
                        assets,
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME)
                makeButtonVisible()
            } catch (e: Exception) {
                throw RuntimeException("Error initializing TensorFlow!", e)
            }
        })
    }

    override fun onBackPressed() {
        super.onBackPressed()
        finish()
    }

    private fun makeButtonVisible() {
        runOnUiThread { button_detect_object!!.visibility = View.VISIBLE }
    }

    private fun displayImageRecognitionResult(results: List<Classifier.Recognition>) {
        results_information_container.visibility = View.VISIBLE
        results_container.removeAllViews()

        val inflater = getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        var counter = 1
        if (results.isNotEmpty()) {
            for (recognizedImage in results) {
                val childView = inflater.inflate(R.layout.view_image_recognition_result, results_container, false)

                val resultTextView = childView.findViewById(R.id.text_view_result) as TextView
                val gtSearchButton = childView.findViewById(R.id.gt_search) as ImageView
                val ecgSearchButton = childView.findViewById(R.id.ecg_search) as ImageView

                gtSearchButton.setOnClickListener({ callGTSearch(recognizedImage) })
                ecgSearchButton.setOnClickListener({ callECGSearch() })

                val text = recognizedImage.title ?: ""
                resultTextView.text = "$counter. $text"
                results_container.addView(childView)
                counter++
            }
        } else {
            val noResultsView = inflater.inflate(R.layout.view_empty_image_recognition_result, results_container, false)
            results_container.addView(noResultsView)
        }
    }

    private fun callGTSearch(recognizedImage: Classifier.Recognition) {
        val resultIntent = Intent()

        val value = rand.nextInt((MAX_PRICE - MIN_PRICE) + 1) + MIN_PRICE

        resultIntent.putExtra("imageText", recognizedImage.title)
        resultIntent.putExtra("imageSuggestedPrice", value.toString())
//        resultIntent.putExtra("imageThumb", picture)
        setResult(Activity.RESULT_OK, resultIntent)
        finish()
    }

    private fun callECGSearch() {
        val ebayIntent = packageManager.getLaunchIntentForPackage("com.ebay.mobile")
        if (ebayIntent != null) {
            startActivity(ebayIntent)
        } else {
            Toast.makeText(this, "Can not find Ebay mobile application on this device", Toast.LENGTH_SHORT).show()
        }
    }
}
