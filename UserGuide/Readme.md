## Feature Extraction Process:

1. Download the pre-processed DEAP dataset  in python format (https://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/data_preprocessed_python.zip).(refer https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html for dataset details) 
2. Set 'data_set_path' under main section pointing to the downloaded pre-processed DEAP dataset
3. Set 'Features_data_file' in line 29, pointing to the desired output path.
4. Execute feature extraction script. (python "deapfeatureextractor.py")
5. The feature extractor script generates features for all the EEG and Phsyiological signals.
6. Delete the unwanted features and keep only features related to EOG and Plethysmograph.
7. The 'participant_questionnaire' (https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html#p_q) contains information w.r.t gender of the participant
8. Combine the features csv and gender information to get the final csv to be used.
9. Use s02, s04, s05, s09, s15, s20, s23, s28, s29 and s30 subjects as test set and remaining as training set.

## Using the models
1. Convert the csv generated in feature extraction process to '.arff' format. (Import the csv into Weka and save it in '.arff' format. Refer https://www.wikihow.tech/Convert-CSV-to-ARFF)
2. Convert Gender and class variable column to nominal data type.
3. The models under '2_class_arousal', '2_class_valence', '4_class' folder are built using Weka software
4. The models under 'MultiLabel', 'MultiTarget' folder are built using Meka Sofware.
5. Import the model into the Weka/Meka software
6. The model parameters can be viewed in the respective software
7. The results mentioned in "Unveiling the Influence of Modelling Approach and Gender in Subject Independent Multimodal Emotion Recognition using EOG and PPG" can be reproduced.
