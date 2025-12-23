# Can you break the captch

This project implements deep learning models to solve two distinct CAPTCHA challenges: Word Classification and Optical Character Recognition (OCR).

**Notebooks**
* **Task 1 (Classification):** [Open in Google Colab](https://colab.research.google.com/drive/1gm24zgsXfFV2YwPx_G7VM4BYjkNxdNAK?usp=sharing)
* **Task 2 (OCR/Generation):** [Open in Kaggle](https://www.kaggle.com/code/sauravroy4245/captchaocr)

---

## Task 1: Classification

### Dataset Generation
The dataset was created using the [Text Recognition Data Generator (TRDG)](https://github.com/Belval/TextRecognitionDataGenerator). This tool generates synthetic text images by applying randomized fonts, backgrounds, and distortions (such as skewing and blurring) to a source list of words.

* **Implementation:** `datasetGenration/GenrateDatasetClassification.py`
* **Word Source:** A subset of 100 words (length > 3) was selected from the [Google-10000-English-No-Swears](https://github.com/first20hours/google-10000-english) list.
* **Fonts:** Fonts were randomly selected from a local collection. All used fonts are available in `datasetGenration/available_fonts`.

### Model Architecture
A custom Deep CNN architecture was designed for the classification task.
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** AdamW with weight decay
* **Model Details:** `classification/ConvmoduleForClassification.py`
* **Training Notebook:** `classification/Task1Classification.ipynb`

### Data Splitting Strategy
For the first iteration, a total of 15,000 samples were generated (5,000 per difficulty category: Easy, Hard, and Bonus).
* **Bonus Set Handling:** To ensure balanced training, the split strategy explicitly handled the "Green" and "Red" subsets within the Bonus category. Both the training and validation splits contain an equal representation of Green and Red images to prevent bias.

---

## Task 2: OCR and Text Generation

### Dataset Generation
The dataset was generated using TRDG. A different word list was used to ensure the Task 2 dataset is disjoint from Task 1 (i.e., the model encounters new words it did not see during the classification task).

### Transfer Learning
To improve performance, the CNN encoder trained in Task 1 was reused for Task 2. By initializing the OCR model with these pre-trained weights, the model leverages previously learned image semantics, resulting in significantly faster convergence.

### Model Architecture
A CRNN (Convolutional Recurrent Neural Network) architecture was implemented for this task.
* **Structure:** Pre-trained CNN (from Task 1) + Bidirectional LSTM + Fully Connected Layer.
* **Loss Function:** CTC (Connectionist Temporal Classification) Loss was used to handle variable-length text sequences.

### Data Splitting Strategy
The same balanced splitting strategy from Task 1 was applied here to ensure fair evaluation across all difficulty levels and variations.

### Words list
[dwyl/english-words](https://github.com/dwyl/english-words) was used with word length > 3 and special character filtered 
see `datasetGenration/GenrateDataset.py` for more details