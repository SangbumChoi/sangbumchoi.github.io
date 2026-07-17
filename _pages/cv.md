---
layout: jarvis-content
title: "CV"
permalink: /cv/
eyebrow: "DANIEL OS / EXPERIENCE"
dek: "AI research, multimodal systems, open-source engineering, and production delivery across six-plus years."
author_profile: false
redirect_from:
  - /resume
---

{% include base_path %}

Summary
======
AI research and systems engineer with 6+ years of experience in multimodal pre-training and post-training, data curation, evaluation, open-source model integration, and production deployment. Built PyTorch training/serving systems and made 40+ contributions across the Hugging Face ecosystem, including 28 public pull requests authored in Transformers.

Resume downloads
======
* [Machine Learning Engineer Resume - Clean Version (PDF)](/files/resume/daniel_choi_resume_clean.pdf)
* [Software Engineer Resume - Long Version (PDF)](/files/resume/daniel_choi_resume_long.pdf)
* [Clean LaTeX source](/files/resume/daniel_choi_resume_clean_tex.zip) / [Long LaTeX source](/files/resume/daniel_choi_resume_long_tex.zip)

Open source
======
* Hugging Face ecosystem contributor with 40+ contributions, including 28 public pull requests authored in Transformers across model architectures, processors, conversion scripts, distributed training fixes, tests, and documentation.
* Hugging Face Transformers contributor: led the addition of Segment Anything 2 (SAM2) support to `huggingface/transformers`.
  * Implemented and refined image/video segmentation model support, processors, documentation, conversion flow, and integration tests through a long-running community review cycle.
  * PR: <a href="https://github.com/huggingface/transformers/pull/32317">Add Segment Anything 2 (SAM2)</a>; merged 2025-08-13.
  * Documentation: <a href="https://huggingface.co/docs/transformers/model_doc/sam2">SAM2 model docs</a>, credited as a model contributor.
* Hugging Face Transformers contributor: opened <a href="https://github.com/huggingface/transformers/pull/43451">Add Molmo2</a> and published <a href="https://huggingface.co/danelcsb/Molmo2-4B">danelcsb/Molmo2-4B</a> on the Hugging Face Hub.

Education
======
* M.S. in Electrical Engineering and Entrepreneurship, Korea Advanced Institute of Science and Technology, KAIST, 2020.03-2021.02
* B.S. in Electrical Engineering, Pohang University of Science and Technology, POSTECH, 2015.03-2020.02
* B.S. in Electrical and Computer Engineering, University of Illinois, Urbana and Champaign, UIUC, 2018.01-2018.12

Work experience
======
* 2026/01-Present: Data Scientist
  * <span style="font-weight:bold">Toss Bank (토스뱅크)</span>
  * Building an on-premise agent system with internally deployed LLMs for secure AI-assisted development workflows.
  * Developing AI-based authentication systems for face and ID card verification.
  * Post-trained and evaluated an approximately 1B-parameter vision-language model in an end-to-end document extraction pipeline, reaching 61% exact-match accuracy for automation-ready outputs.

* 2021/09-2026/01: Machine Learning Engineer
  * <span style="font-weight:bold">SuperbAI</span>
  * Led multimodal pre-training and staged text/image alignment for a visual-grounding model using a curated 1.1M-image dataset.
  * Built distributed, multi-GPU training and serving infrastructure with AWS Batch, TensorRT, and Triton Inference Server, improving inference throughput by 5x over pure PyTorch serving.
  * Built LoRA, Adapter, and LST post-training pipelines and delivered 1,100+ models and 60+ customer endpoints in one year.
  * Built interactive segmentation tools using RepViT-SAM, FocalClick, SAM, and SAM2, increasing segmentation labeling speed by 25x.
  * Reduced GPU memory usage by 65.6% and training time by 44.3% using parameter- and memory-efficient training methods.
  * Won 2nd place in IOD and 4th place in FSOD challenges at CVPR 2025.

* 2021/02-2021/08: Machine Learning Engineer Intern
  * <span style="font-weight:bold">Kakao Enterprise</span>
  * Fixed AutoGluon NeuralNetFastAI scaling issue.
  * Developed a Flask-based training and inference AutoML framework with a simple front-end.

* 2019/02-2020/06: Co. Team Island CTO
  * <span style="font-weight:bold">Team Island</span>
  * Built [ZZAZZ (째즈)](https://www.venturesquare.net/821623), a mobile video-editing application that applied customizable motion effects to people using detection/segmentation, 3D mapping, and tracking.
  * Built lightweight CNN models for mobile applications and on-device inference.

* 2018/08-12: Undergraduate Researcher
  * <span style="font-weight:bold">UIUC Undergraduate Research Program</span>
  * Improved direction-of-arrival estimation with the MUSIC algorithm using irregular microphone arrays.
  * Generated binaural sounds through a software-based audio pipeline.
  * Supervisor: Professor <a href="https://synrg.csl.illinois.edu/">Romit Roy Choudhury</a>

* 2018/06-08: Machine Learning Engineer
  * <span style="font-weight:bold">Seerslab Intern</span>
  * Developed face landmark detection using Haar cascades, HOG features, and machine learning methods.
  * Built a GUI tool for annotating face coordinates.
  * Developed CMS login functionality with JWT-based authentication.

* 2017/03-06: Undergraduate Researcher
  * <span style="font-weight:bold">POSTECH Undergraduate Research Program</span>
  * Developed a non-invasive heart-rate measurement device.
  * Collaborated on a smart-watch FPGA module for a national research project.
  * Supervisor: Professor <a href="https://postechimslab.wixsite.com/citeimslab">Park Sung Min</a>

* 2016/06-08: Research Intern
  * <span style="font-weight:bold">ASAN Medical Center</span>
  * Designed simulations for an electrical surgical unit using electromagnetic field analysis tools.
  * Supported development and experiments for an assistive device for knee-injured patients.
  * Supervisor: Professor Choi Jae Soon
  
Skills
======
* Research: PyTorch, Hugging Face Transformers, multimodal pre-training and post-training, LoRA, contrastive alignment, evaluation
* AI systems: AWS Batch, distributed multi-GPU training, TensorRT, Triton Inference Server, MLflow, model conversion
* Programming: Python, Kotlin, Git; large open-source codebase development, testing, and documentation
* Deployment: mobile ML, lightweight CNNs, on-device inference, TensorFlow Lite
* AI developer tools: Claude Code, Codex, opencode, LangChain, Langfuse
* Hardware and tools: FPGA development with Xilinx, PyQt5
* Languages: Korean (native), English (professional)

Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Athlete Service and Leadership
======
* CTO, Team Island, 2019/02-2020/06
* Open-source contributor to Hugging Face Transformers
