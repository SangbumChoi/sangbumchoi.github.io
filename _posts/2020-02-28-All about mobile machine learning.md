---
title: 'All About Mobile Machine Learning'
date: 2020-02-28
permalink: /posts/2020/02/All about mobile machine learning/
tags:
  - Study
  - Mobile Application
  - TF Lite
---

# TensorFlow Lite and CoreML

![TFlite architecture](/images/TFLite_Architecture.png)

Tensorflow lite has 5 different compoenents to transform. Basically you can transform original model that you build on computer. Java/C++ API works in Android and C++ API works in iOS. For the hardware acceleration, device can use Neural Networks API if it is available.

Tensorflow lite uses direct graphic acceleratation, Open Graphics Library (OpenGL) on Android
and Metal on iOS. 

**What is OpenGL?**
OpenGL is a standard API unit for 2 and 3 dimension graphics used for supporting
multi programming langauge in diverse platforms. Not only for regularization but also has pipelining method contains inside of this API.

![Kera to Android](/images/tf_Keras_to_Android.png)

## TensorFlow Lite on Android 

We can start using the demo app provided in the TensorFlow GitHub repository.
[Android Tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android) followed by this page, this page describes camera application that classifies input image from camera using inception-v3 model or a quantized mobilenet model.

There are three way to make using Tensorflow as mobile application
* Using pre-build APK file (direct downloading)
* Build on own Android Studio and run the application
* Download Bazel to download the source code of TensorFlow liste and run on the app through the command line

We can think of TensorFlow Lite is kind of library tool that can be used in our application development.

```
NOTE: Bazel does not currently support building for Android on Windows. Full support for gradle/cmake builds is coming soon, but in the meantime we suggest that Windows users download the prebuilt demo APK instead.
```
Above notion is still on the tensorflow github

**What is Bazel?**

According to official site in Bazel homepage, it describes itself as a fast, and stable build tool for any size of software.
Most famous build tool for example we can say 'Gradle' which is come from android studio.
Google said "If you want to build a lot of code and various languages for one project or have to make multi-platform then bazel should be useful"

## ARFaceDetection Tutorial
This tutorial is based on github pages from Cleveroad.

![Demo](/images/demo_Cleveroad.gif)

For initial Setup in this tutorial you need to add dependencies with firebase and ARFaceDetection

```groovy
ARFaceDetection-master/ar-face-detection/build.gradle

apply plugin: 'com.android.application'

apply plugin: 'kotlin-android'

apply plugin: 'kotlin-android-extensions'

apply plugin: 'kotlin-kapt'

...

dependencies {
    implementation 'com.google.firebase:firebase-core:16.0.9'
    implementation 'com.cleveroad.ARFaceDetection:ar-face-detection:1.0.2'
}
```

Firebase
*   Create a Firebase project to connect to your Android app and setup it.
    Link to firebase console https://console.firebase.google.com/ 
*   Move your config file(google-services.json) into the module (app-level) directory of your app.

Reference

[TensorflowLite with mobile application](https://soundlly.github.io/2017/11/20/tensorflowlite-moblienet-demo/)

[https://www.tensorflow.org/lite/performance/gpu](https://www.tensorflow.org/lite/performance/gpu)

[Using Firebase](https://firebase.google.com/docs/ml-kit/android/detect-objects)

Github

[ARFaceDetection](https://github.com/Cleveroad/ARFaceDetection)