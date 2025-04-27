# assignment-1-image-classification-knn-svm-softmax-neural-network-solved
**TO GET THIS SOLUTION VISIT:** [Assignment #1: Image Classification, kNN, SVM, Softmax, Neural Network Solved](https://www.ankitcodinghub.com/product/assignment-1-image-classification-knn-svm-softmax-neural-network-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;39977&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Assignment #1: Image Classification, kNN, SVM, Softmax, Neural Network Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

<ul>
<li>understand the basic&nbsp;<strong>Image Classification pipeline</strong>&nbsp;and the data-driven approach (train/predict stages)</li>
<li>understand the train/val/test&nbsp;<strong>splits</strong>&nbsp;and the use of validation data for&nbsp;<strong>hyperparameter tuning</strong>.</li>
<li>develop proficiency in writing efficient&nbsp;<strong>vectorized</strong>&nbsp;code with numpy</li>
<li>implement and apply a k-Nearest Neighbor (<strong>kNN</strong>) classifier</li>
<li>implement and apply a Multiclass Support Vector Machine (<strong>SVM</strong>) classifier</li>
<li>implement and apply a&nbsp;<strong>Softmax</strong>&nbsp;classifier</li>
<li>implement and apply a&nbsp;<strong>Two layer neural network</strong>&nbsp;classifier</li>
<li>understand the differences and tradeoffs between these classifiers</li>
<li>get a basic understanding of performance improvements from using&nbsp;<strong>higher-level representations</strong>&nbsp;than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)</li>
</ul>
<h2 id="setup">Setup</h2>
Get the code as a zip file&nbsp;<a href="http://cs231n.github.io/assignments/2019/spring1819_assignment1.zip">here</a>.

You can follow the setup instructions&nbsp;<a href="http://cs231n.github.io/setup-instructions">here</a>.

<h3 id="download-data">Download data:</h3>
Once you have the starter code (regardless of which method you choose above), you will need to download the CIFAR-10 dataset. Run the following from the&nbsp;<code class="highlighter-rouge">assignment1</code>&nbsp;directory:

<div class="language-bash highlighter-rouge">
<div class="highlight">
<pre class="highlight"><code><span class="nb">cd </span>cs231n/datasets
./get_datasets.sh
</code></pre>
</div>
</div>
<h3 id="start-ipython">Start IPython:</h3>
After you have the CIFAR-10 data, you should start the IPython notebook server from the&nbsp;<code class="highlighter-rouge">assignment1</code>&nbsp;directory, with the&nbsp;<code class="highlighter-rouge">jupyter notebook</code>&nbsp;command. (See the&nbsp;<a href="https://github.com/cs231n/gcloud/">Google Cloud Tutorial</a>&nbsp;for any additional steps you may need to do for setting this up, if you are working remotely)

If you are unfamiliar with IPython, you can also refer to our&nbsp;<a href="http://cs231n.github.io/ipython-tutorial">IPython tutorial</a>.

<h3 id="some-notes">Some Notes</h3>
<strong>NOTE 1:</strong>&nbsp;There are&nbsp;<code class="highlighter-rouge"># *****START OF YOUR CODE</code>/<code class="highlighter-rouge"># *****END OF YOUR CODE</code>&nbsp;tags denoting the start and end of code sections you should fill out. Take care to not delete or modify these tags, or your assignment may not be properly graded.

<strong>NOTE 2:</strong>&nbsp;The submission process this year has&nbsp;<strong>2 steps</strong>, requiring you to 1. run a submission script and 2. download/upload an auto-generated pdf (details below.) We suggest&nbsp;<strong><em>making a test submission early on</em></strong>&nbsp;to make sure you are able to successfully submit your assignment on time (a maximum of 10 submissions can be made.)

<strong>NOTE 3:</strong>&nbsp;This year, the&nbsp;<code class="highlighter-rouge">assignment1</code>&nbsp;code has been tested to be compatible with python version&nbsp;<code class="highlighter-rouge">3.7</code>&nbsp;(it may work with other versions of&nbsp;<code class="highlighter-rouge">3.x</code>, but we won’t be officially supporting them). You will need to make sure that during your virtual environment setup that the correct version of&nbsp;<code class="highlighter-rouge">python</code>&nbsp;is used. You can confirm your python version by (1) activating your virtualenv and (2) running&nbsp;<code class="highlighter-rouge">which python</code>.

<strong>NOTE 4:</strong>&nbsp;If you are working in a virtual environment on OSX, you may&nbsp;<em>potentially</em>&nbsp;encounter errors with matplotlib due to the&nbsp;<a href="http://matplotlib.org/faq/virtualenv_faq.html">issues described here</a>. In our testing, it seems that this issue is no longer present with the most recent version of matplotlib, but if you do end up running into this issue you may have to use the&nbsp;<code class="highlighter-rouge">start_ipython_osx.sh</code>&nbsp;script from the&nbsp;<code class="highlighter-rouge">assignment1</code>&nbsp;directory (instead of&nbsp;<code class="highlighter-rouge">jupyter notebook</code>&nbsp;above) to launch your IPython notebook server. Note that you may have to modify some variables within the script to match your version of python/installation directory. The script assumes that your virtual environment is named&nbsp;<code class="highlighter-rouge">.env</code>.

<h3 id="q1-k-nearest-neighbor-classifier-20-points">Q1: k-Nearest Neighbor classifier (20 points)</h3>
The IPython Notebook&nbsp;<strong>knn.ipynb</strong>&nbsp;will walk you through implementing the kNN classifier.

<h3 id="q2-training-a-support-vector-machine-25-points">Q2: Training a Support Vector Machine (25 points)</h3>
The IPython Notebook&nbsp;<strong>svm.ipynb</strong>&nbsp;will walk you through implementing the SVM classifier.

<h3 id="q3-implement-a-softmax-classifier-20-points">Q3: Implement a Softmax classifier (20 points)</h3>
The IPython Notebook&nbsp;<strong>softmax.ipynb</strong>&nbsp;will walk you through implementing the Softmax classifier.

<h3 id="q4-two-layer-neural-network-25-points">Q4: Two-Layer Neural Network (25 points)</h3>
The IPython Notebook&nbsp;<strong>two_layer_net.ipynb</strong>&nbsp;will walk you through the implementation of a two-layer neural network classifier.

<h3 id="q5-higher-level-representations-image-features-10-points">Q5: Higher Level Representations: Image Features (10 points)</h3>
The IPython Notebook&nbsp;<strong>features.ipynb</strong>&nbsp;will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.
