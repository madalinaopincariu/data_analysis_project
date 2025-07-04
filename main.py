from src.clustering.clustering import kmeansclustering
from src.preprocessing.data_processing import clean_and_normalize_data
from src.visualizations.data_visualizations import create_visualizations


def main():

    input_data = 'data/raw/depression_data.csv'
    output_data = 'data/cleaned/cleaned_data.csv'
    interpreted_data_path = 'data/interpreted/interpreted_data.csv'
    clean_and_normalize_data(input_data, output_data)
    #create_visualizations(output_data, input_data)
    kmeansclustering(output_data,interpreted_data_path)

if __name__ == "__main__":
    main()