The definition and training process of the model are illustrated in [matron2017\wakeword-detection-model](git@github.com:matron2017/Real-time-wake-word-detection.git)

The model architecture is following: 
```bash
1D CRNN Keyword Spotting Model Architecture:

-------------------------------------------------
Input: (301, 48)
|
|--- Conv1D (layer1): num_units=48, kernel_size=3, padding="same", activation="relu"
|--- Dropout: dropout_ratio=0.3
|
|--- Conv1D (layer2): num_units=48, kernel_size=3, padding="same", activation="relu"
|--- Dropout: dropout_ratio=0.3
|
|--- GRU (RNN_1): num_units=48, return_sequences=False
|
|--- Dense (dense_a): num_units=2, activation="softmax"
-------------------------------------------------
```

