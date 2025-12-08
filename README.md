# CS255 Computer Security - Lab 4: Adversarial Learning (Fall 2025)

## Overview

In this lab, you will learn how to generate adversarial examples to attack a CNN deep-learning model. Specifically, you will generate a set of adversarial images that cause the model to misclassify them. 

Project is based on this paper 
[Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-4_Xu_paper.pdf)

---

## Prerequisites

- Python with [PyTorch](https://pytorch.org) and `torchvision` installed.  
- (Optional but recommended) A machine with NVIDIA GPU + CUDA support for faster inference.  
  If you don’t have a GPU, you can use [Google Colab](https://colab.research.google.com) which provides a free Nvidia T4.

In addition, download the following provided files:

1. **`testset.pt`** - Contains 100 images and their correct labels in the form `(input_imgs, labels)`.  
   - Each image tensor has shape `3 × 32 × 32` (channels × height × width).  
   - Each label is an integer (class index).  
   - These are the original clean images that you will perturb.  

2. **`resnet.model`** - A pre-trained ResNet classification model.  
   - The model accepts an image tensor input and outputs a 1000-dimensional vector of class probabilities.

---

## Algorithms

You will implement two adversarial attack algorithms:

### Fast Gradient Sign Method (FGSM) - 10 pts

- Use the FGSM method to generate adversarial examples.  
- For each input image, compute the gradient of the loss with respect to the input, take the sign, and perturb by ε:  


x_adv = x + ε * sign(∇_x L(model(x), y))


- Save the 100 resulting adversarial images (in the same order as original) using `torch.save()`.

Implementation tips:  
- Load the provided model and dataset.  
- Use `torch.utils.data.DataLoader` or a simple loop to iterate the test set.  
- Use `model(input_imgs)` to obtain predictions.

### Projected Gradient Descent (PGD) - 2 pts (bonus)

- An iterative extension of FGSM. In each iteration:  
1. Compute gradient of loss w.r.t input.  
2. Take a small step in the gradient sign direction.  
3. Project the perturbed image back into the `L_∞` ball of radius ε around the original image.  
- Repeat for a fixed number of iterations.  
- Save resulting adversarial images similar to FGSM.

---

## Submission Format

- Your final submission (for FGSM or PGD) should be a list of 100 `torch.Tensor`, each of shape `3 × 32 × 32`, in the same ordering as the original test set.  
- Save using e.g.  

```python
torch.save(adv_imgs, "fgsm.pt")
````

or

```python
torch.save(adv_imgs, "pgd.pt")
```

* Upload `fgsm.pt` (and optionally `pgd.pt`) through Gradescope. The evaluation is automated.
* **Do not** submit incorrect or malformed files — this may result in zero or very low grade. It is recommended you build a “self-check” script to verify before submission.

---

## Grading Policy

Your adversarial samples are scored based on three metrics:

* **ε** - The average per-pixel difference between adversarial and original images. Smaller ε is better (less visible perturbation).
* **`tp1_accuracy_diff`** - The drop in Top-1 accuracy between original and adversarial set.
* **`tp5_accuracy_diff`** - The drop in Top-5 accuracy.

### Score formulas

* **FGSM (max 10 pts):**

  ```
  TP1_Score = 0.33 + 33 × tp1_accuracy_diff  
  TP5_Score = 66 × tp5_accuracy_diff  
  Final_FGSM_Score = min( max( max(TP1_Score, TP5_Score), 0 ), 10 )
  ```
* **PGD (max 2 pts bonus):**

  ```
  TP1_Score = 26 × tp1_accuracy_diff  
  TP5_Score = 46 × tp5_accuracy_diff  
  Final_PGD_Bonus = 0.2 × min( max( max(TP1_Score, TP5_Score), 0 ), 10 )
  ```

Thus, to maximize your grade:

* Produce strong misclassification (large accuracy drop)
* While keeping perturbation ε as small as possible

---

## Results from Gradescope

<pre>
Grading FGSM:
Top-1 error on original samples:  40.00%; on adversarial samples: 82.00%
Top-5 error on original samples:  9.00%; on adversarial samples: 28.00%
Accuracy drop: tp1_accuracy_diff = 0.4200, tp5-accuracy-diff = 0.1900, eps = 0.049330
[FGSM] tp1_score=14.1900, tp5_score=12.5400, final FGSM score=10.0000


Grading PGD:
Top-1 error on original samples:  40.00%; on adversarial samples: 95.00%
Top-5 error on original samples:  9.00%; on adversarial samples: 43.00%
Accuracy drop: tp1_accuracy_diff = 0.5500, tp5-accuracy-diff = 0.3400, eps = 0.065308
[PGD] tp1_score=14.3000, tp5_score=15.6400, final PGD bonus=2.0000
</pre>
