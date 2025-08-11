# **Data Wrangling and Preprocessing**

This stage turns a raw collection of classical piano MIDIs into a clean, balanced, and model-ready dataset. We remove duplicates, clean each file, rebalance classes, and extract features used by the CNN and LSTM models. The output is a cached folder of features plus a manifest the training notebooks can load quickly:


* Gather and de-duplicate MIDIs per composer, then stage a local working copy.
* Clean each file: drop drums and empty tracks, fix note bounds to A0-C8, sanitize velocities and durations, and skip corrupted or extreme files.
* Verify counts so only unique, valid pieces move forward.
* Rebalance the dataset with light augmentation on minority classes: pitch shift, time stretch, and velocity jitter, while keeping originals.
* Extract model features for every piece:
  * A fixed 88x512 piano-roll for the CNN
  * An event-token sequence for the LSTM
  * Chord labels for analysis
* Save features and a manifest CSV so training notebooks can load data fast and reproducibly.


# **CNN modeling for composer identification**

In CNN Modeling stage we build a convolutional model that predicts the composer from a fixed piano-roll window. A CNN learns local time-pitch patterns (motifs, cadences, voicings) and combines them across the roll to form a global decision.

**Why LSTM is useful for our case**

Convolutions capture short, repeating shapes in the piano-roll: intervals stacked in pitch, note runs in time, and their combinations. Pooling lets the model generalize across position, so the same pattern recognized anywhere in the window contributes to the prediction.

**How we use it with MIDI**

We work with the piano-roll features produced in the data wrangling notebook. Each piece is a binary roll with 88 keys across 512 frames. We treat it like a single-channel image of shape `(512, 88, 1)` and apply 2D convolutions over time and pitch.

**Notebook Steps:**

* Load the manifest and the cached `.pr.npy` piano-rolls.
* Build the dataset:
  * ensure shape `(512, 88, 1)` and batch size
  * train/validation/test splits from the saved CSVs
  * prefetch for throughput
* Define the CNN:
  * stacks of Conv-BN-ReLU with pooling over time and pitch
  * global average pooling to get one vector per roll
  * a small dense head with dropout and a softmax over the composers
* Train with cross-entropy loss and Adam. Track validation accuracy. Use early stopping and checkpoints.
* Evaluate: accuracy, per-class report, and a confusion matrix.
* Save the trained `.keras` model, the label map, and the preprocessing notes for inference in the app.


# **LSTM modeling for composer identification**

In LSTM Modeling stage we build a sequence model that predicts the composer from a stream of MIDI events. An LSTM is a recurrent neural network that keeps track of what came before. That makes it a good fit for MIDI, where the order and timing of events matter.

**Why LSTM is useful for our case**

An LSTM reads tokens one by one and updates an internal memory. It can learn patterns like typical intervals, cadences, and rhythmic habits that unfold across time.

**How we use it with MIDI**

We work with the event features produced in the data wrangling notebook. Each piece is a list of tokens drawn from a small vocabulary: time-shift steps, note-on, and note-off. Events are already quantized to fixed time steps and trimmed to a maximum length.

**Notebook Steps:**

* Load the manifest and the event token files.
* Build the dataset: pad or truncate sequences to a fixed length, create train and validation splits, and batch them.
* Define the model: token embedding, one or more LSTM layers, a pooled sequence representation, and a softmax layer over the composers.
* Train with cross-entropy loss and Adam. Track validation loss and accuracy. Use early stopping and model checkpoints.
* Evaluate: report accuracy per composer and a confusion matrix.
* Save the trained weights, the label map, and the exact preprocessing settings.
