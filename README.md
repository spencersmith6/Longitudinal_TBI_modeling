# Longitudinal_TBI_modeling
Exploration in how to classify different longitudinal curves. The data is simulated evaluation of tramatic brain injury victims. Each victim has between 6 and 30 assessment observations at different times.

Phase One: 
The patient evaluations all happend at different times after the brain injury. In order to compare the feature level at each observation the data must be aggregated into buckets at standard time intervals. The following report describes the techniques used.

<a href="reports/Research_progress_on_simulated_data_(2-5-17).pdf">Phase one- Bucketing Techniques</a>


Phase Two:
The ultimate goal is to classify the different types of TBI's and TBI recovery progress using the longtudinal curves/trajectories. A method to acomplish this is to fit polynomial curves to each victims recovery trajectory. Then perform machine learning classification on the coefficients of those curves. The following report describes the techniques used.

<a href="reports/Research_progress_(2-12-17).pdf">Phase one- Bucketing Techniques</a>

