# MPBD-LSTM: A Predictive Model For Colorectal Liver Metastases Using Time Series Multi-phase Contrast-Enhanced CT Scans
This is the official repo of the article "MPBD-LSTM: A Predictive Model For Colorectal Liver Metastases Using Time Series Multi-phase Contrast-Enhanced CT Scans" (MICCAI 2023). Code will be uploaded shortly.

## Dataset:
Our dataset consists of 269 patients in total, with a positive rate of 25.7%. Aside from the plain scan phase, it has 2 additional contrast-enhanced phase: the portal venous phase (V) and the arterial phase.
It's collected based on the following rules:<br /> 
* No tumor appears on the CT scans. That means patients have not been
diagnosed as CRLM when they took the scans.
* Patients were previously diagnosed with colorectal cancer TNM stage I to
stage III, and recovered from colorectal radical surgery.
* Patients have two or more times of CECT scans.
* We already determined whether or not the patients had liver metastases
within 2 years after the surgery, and manually labeled the dataset based on
this.
* No potential focal infection in the liver before the colorectal radical surgery.
* No metastases in other organs before the liver metastases.
* No other malignant tumors.


For more details, please refer to the article. <br />
We are further doing more experiments for the task, will upload the code once it's finished.
Please send email to the author at xli34@nd.edu for the link and password to download the dataset. 
