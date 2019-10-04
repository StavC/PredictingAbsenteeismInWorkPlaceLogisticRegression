from absenteeism_module import *


def main():

    model=absenteeism_model('model','scaler')
    model.load_and_clean_data('Absenteeism_new_data.csv')
    print(model.predicted_outputs())
    #Probaility-the probaility that a give individual is expected to be absent from work for more than 3 hours(median)
    #Prediction 1 if the probaility is 50% or higher 0 if lower





if __name__ == '__main__':
    main()