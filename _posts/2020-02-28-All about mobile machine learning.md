---
title: 'All about mobile machine learning'
date: 2020-02-28
permalink: /posts/2020/02/All about mobile machine learning/
tags:
  - Computer Vision
  - Machine Learning
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

Further filled

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

Github

[ARFaceDetection](https://github.com/Cleveroad/ARFaceDetection)