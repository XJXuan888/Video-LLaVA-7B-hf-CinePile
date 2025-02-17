
# CinePile fine tune

![CinePile Benchmark - Open vs Closed models](https://huggingface.co/mfarre/Video-LLaVA-7B-hf-CinePile/resolve/main/benchmark.png)

## Table of Contents
1. [Introduction](#introduction)
2. [What Can You Find in This Repo](#what-can-you-find-in-this-repo)
3. [Results](#results)
4. [Model Sources](#model-sources)
5. [Uses](#uses)
6. [Setup and Installation](#setup-and-installation)

---

## Introduction
This repository provides tools and resources for fine-tuning the [Video-LLaVA](https://huggingface.co/docs/transformers/main/en/model_doc/video_llava) model on the [CinePile](https://huggingface.co/datasets/tomg-group-umd/cinepile) dataset, a benchmark for long-form video understanding. Video-LLaVA is an open-source multimodal model capable of processing both images and videos, making it ideal for tasks like answering multiple-choice questions based on video content.

The fine-tuned model is optimized for CinePile-related queries, such as:
- **Temporal aspects** (e.g., "When did the event happen?")
- **Character and relationship dynamics** (e.g., "How do the characters interact?")
- **Narrative and plot analysis** (e.g., "What is the main conflict?")
- **Theme exploration** (e.g., "What is the underlying message of the story?")

By fine-tuning Video-LLaVA, we aim to bridge the performance gap between open and closed models, achieving results comparable to state-of-the-art models like Claude 3 (Opus).

## What can you find in this repo
- **Fine-Tuning Notebook**: A Jupyter Notebook to reproduce the fine-tuning task on Video-LLaVA.
- **Inference Notebook**: A Jupyter Notebook to run inference using our model weights on wild images and the CinePile dataset.
- **Dataset Configuration**: A `video2dataset` configuration to download the CinePile dataset.

---

## Results
Video multimodal research often emphasizes activity recognition and object-centered tasks, such as determining "what is the person wearing a red hat doing?" However, this focus overlooks areas like theme exploration, narrative and plot analysis, and character and relationship dynamics.
CinePile addresses these areas in their benchmark and they find that Large Language Models significantly lag behind human performance in these tasks. Additionally, there is a notable disparity in performance between open and closed models.

In our initial fine-tuning, our goal was to assess how well open models can approach the performance of closed models. By fine-tuning Video LlaVa, we achieved performance levels comparable to those of Claude 3 (Opus).

| Model                          | Average | Character and relationship dynamics | Narrative and Plot Analysis | Setting and Technical Analysis | Temporal | Theme Exploration |
|--------------------------------|---------|-------------------------------------|-----------------------------|--------------------------------|----------|-------------------|
| Human                          | 73.21   | 82.92                               | 75                          | 73                             | 75.52    | 64.93             |
| Human (authors)                | 86      | 92                                  | 87.5                        | 71.2                           | 100      | 75                |
| GPT-4o                     | 59.72   | 64.36                               | 74.08                       | 54.77                          | 44.91    | 67.89             |
| GPT-4 Vision                | 58.81   | 63.73                               | 73.43                       | 52.55                          | 46.22    | 65.79             |
| Gemini 1.5 Pro             | 61.36   | 65.17                               | 71.01                       | 59.57                          | 46.75    | 63.27             |
| Gemini 1.5 Flash           | 57.52   | 61.91                               | 69.15                       | 54.86                          | 41.34    | 61.22             |
| Gemini Pro Vision           | 50.64   | 54.16                               | 65.5                        | 46.97                          | 35.8     | 58.82             |
| Claude 3 (Opus)            | 45.6    | 48.89                               | 57.88                       | 40.73                          | 37.65    | 47.89             |
| **Video LlaVa - this fine-tune**       | **44.16**   | **45.26**                              | **45.14**                       | **46.93**                          | **32.55**    | **49.47**         |
| Video LLaVa               | 22.51   | 23.11                               | 25.92                       | 20.69                          | 22.38    | 22.63             |
| mPLUG-Owl                 | 10.57   | 10.65                               | 11.04                       | 9.18                           | 11.89    | 15.05             |
| Video-ChatGPT            | 14.55   | 16.02                               | 14.83                       | 15.54                          | 6.88     | 18.86             |
| MovieChat                | 4.61    | 4.95                                | 4.29                        | 5.23                           | 2.48     | 4.21              |




Fine-tuned model taking as bases [Video-LlaVA](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf) to evaluate its performance on CinePile.



## Model Sources

[Hugging Face](https://huggingface.co/mfarre/Video-LLaVA-7B-hf-CinePile) model card and weights

---

## Uses
Although the model can answer questions based on the content, it is specifically optimized for addressing CinePile-related queries.
When the questions do not follow a CinePile-specific prompt, the inference section of the notebook is designed to refine and clean up the text produced by the model.

## Setup and Instructions
Each notebook includes detailed instructions at the top to guide you through the process:

1. **Fine-Tuning Notebook**:
   - Instructions for setting up the environment, loading the dataset, and running the fine-tuning process are provided at the top of the notebook.
   - Follow the step-by-step guide to fine-tune Video-LLaVA on the CinePile dataset.

2. **Inference Notebook**:
   - Instructions for loading the fine-tuned model, running inference on YouTube videos, and evaluating the model on the CinePile test dataset are included at the top.
   - Use the provided examples to test the model on custom inputs or benchmark its performance.