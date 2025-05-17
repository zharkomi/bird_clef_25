# Bird CLEF 25
This is my submission to the Kaggle competition. The goal of this competition is to classify bird species from audio recordings. The dataset consists of a large number of audio files, each containing a recording of a bird's call or song.

https://www.kaggle.com/competitions/birdclef-2025/overview

In this project I heavily relied on Claude to generate code. Classification was done based on Birdnet embeddings. Full training pipeline is the following:
1. Remove human voice from recordings
2. Apply data augmentation to the audio files from species which don't have enough data
3. Extract embedding with Birdnet library
4. Split all data in 5 folds
5. Train model on each fold
6. Generate final model as aggregation of all folds

Inference process:
1. Apply Birdnet lib to a file
2. Get embeddings for each 3 seconds chunk
3. Make prediction with trained model
4. With spline convert 3 seconds chunks to 5 second chunks

What didn't work:
1. Use Birdnet for known species and train model only for new
2. Adding noise with wavelets

This approach improve performance by about 12% compared to pure Birdnet lib usage from 0.610 to 0.731. Probably relying on Birdnet embeddings was not the best idea, looks like it has some limitations.

*Disclaimer: that was my first take into ML*