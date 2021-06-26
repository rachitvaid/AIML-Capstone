# AIML Capstone
## Automated IT Ticket Assignment
This project aims to automate the Assignment of IT tickets to appropriate functional groups. 

### Problem Statement:
[IT Incident Management Process](https://github.com/rachitvaid/AIML-Capstone/blob/master/Documentation/Automatic%20Ticket%20Assignment.pdf)

### Input Data:
The input data consists of following text fields:
| Field | Description |
| ----------- | ----------- |
| Short Description | Brief description of issue |
| Description | Detailed description of issue |
| Caller | User who raised the ticket |
| Assignment group | Group to which ticket was assigned |

### Project Structure
The project is divided into two parts- V1 Interim and V2 Final.  
#### V1 Interim:
This section of the project comprises of the initial few weeks of work. The code contains basic EDA, data cleansing and training various ML models.  
- [Interim Notebook](https://github.com/rachitvaid/AIML-Capstone/blob/master/Code/V1%20Interim/AIML_Capstone_Interim_Notebook.ipynb)  
- [Interim Project Report](https://github.com/rachitvaid/AIML-Capstone/blob/master/Documentation/Reports/AIML%20Capstone%20Interim%20Report.docx)  
- [Interim Presentation](https://github.com/rachitvaid/AIML-Capstone/blob/master/Documentation/Presentations/Automatic%20Ticket%20Assignment%20-%20Interim.pptx)  

#### V2 Final
This section of the project consolidates all the trained models, tunes the hyperparameters for the best model and create a UI application.  
- ##### *Notebooks*
The final project is split into 4 notebooks due to RAM limitations. The proper order of execution is mentioned below.  
1. [AIML_Capstone_N1_EDA_Data_Tuning](https://github.com/rachitvaid/AIML-Capstone/blob/master/Code/V2%20Final/AIML_Capstone_N1_EDA_Data_Tuning.ipynb) -- EDA, Data cleansing and formatting, data visualisation, training of various AI and ML models
2. [AIML_Capstone_N2_LSTM_Models](https://github.com/rachitvaid/AIML-Capstone/blob/master/Code/V2%20Final/AIML_Capstone_N2_LSTM_Models.ipynb) -- Training data on LSTM model
3. [AIML_Capstone_N3_Final_Model_Hyperparameter_Tuning](https://github.com/rachitvaid/AIML-Capstone/blob/master/Code/V2%20Final/AIML_Capstone_N3_Final_Model_Hyperparameter_Tuning.ipynb) -- Hyperparameter tuning for selected model i.e. Neural Networks
4. [AIML_Capstone_N4_Final_model_pickle_generator](https://github.com/rachitvaid/AIML-Capstone/blob/master/Code/V2%20Final/AIML_Capstone_N4_Final_model_pickle_generator.ipynb) -- Pickling the tuned Neural network model for future use.  

- ##### *Files*
Multiple intermediate files are created and used among these notebooks.  
* List of Intermediate files:
  * LR_Results.csv
  * RF_Results.csv
  * word2vec_vector.txt
  * resampled_tickets.pkl
  * cleaned_en_ticket.csv
  * cleaned_ticket_df.csv
  * basic_model_tuning_result.csv
  * cleaned_tuned_vec_ticket.csv

Following files are generated as part of code to be used by UI.  
* List of generated files:
  * tfidf_vec.pkl
  * target_enc_dict.obj
  * Final_Model_Pkl
These files are placed in UI directory and are required for UI app execution.  

- ##### *UI App*
A UI app created using Streamlit can be executed to classify the tickets based on description of issue passed.  
Steps to execute Streamlit:  
1. Install Streamlit using command - pip install streamlit
2. Place the files present in the [UI directory](https://github.com/rachitvaid/AIML-Capstone/tree/master/UI) in a single folder
3. Navigate to the above directory in the terminal
4. Execute command - streamlit run Streamlit.py
  
The web app would run in the default web browser.  

The demo of the App can be seen [here](https://github.com/rachitvaid/AIML-Capstone/blob/master/UI/Demo%20Video/IT%20ticket%20classification%20using%20Streamlit.mp4)

- ##### *Documentation*
The following three files are created as part of project documentation:  
1. [Final Project Report](https://github.com/rachitvaid/AIML-Capstone/blob/master/Documentation/Reports/AIML%20Capstone%20Final%20Report.docx) - The final project report consolidates the whole Project execution in a single document. It is a step by step description of methodology followed.
2. [Final Project Presentation](https://github.com/rachitvaid/AIML-Capstone/blob/master/Documentation/Presentations/Automatic%20Ticket%20Assignment%20-%20Final.pptx) - This is the detailed presentation for technical users.
3. [Project Presentation- Business](https://github.com/rachitvaid/AIML-Capstone/blob/master/Documentation/Presentations/Automatic%20Ticket%20Assignment%20-%20Business.pptx) - This is the overall presentation for business users.
