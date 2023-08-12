#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from argparse import ArgumentParser
from pathlib import Path

dname = Path(__file__).parent

#%%
def buildCropDiseaseCountTuple(instanceFolder):
    # Name of Crop and Disease is the folder name so split it . Need to change the logic here based on the data source . Much Pain!!
    # Count will be just samples within the folder
    str_name = str(instanceFolder.name)
    str_name = str_name.replace(" leaf", "").rstrip()
    print(str_name)
    values = str_name.split(" ", 2)
    f = instanceFolder.rglob('*')
    counts = np.unique([x.parent for x in f], return_counts=True)[1]
    if len(values) == 1:
        #values = ['NoCrop', 'Background']
        values.append('healthy')
    return (values[0], values[1], counts[0].tolist())
    #logging.debug(values, counts[0].tolist())
    
#%%
def readFolderAndSaveDataFrame(DataFolder, csvFilename):
    '''
    Read the folder - Note the assumption here is that the images are already seperated into their respective class subfolders
    Example -
        <root>
            - <class 1>
                - <image 1>
                - <image 2>
            - <class 2>
                - <image 1>
                - <image 2>
    '''
    dataFolder = Path(f"{dname}/{DataFolder}")
    dataList = []
    # Get all the folders within the path - The list of folders are the classes.
    instancePathList = [f for f in dataFolder.iterdir() if f.is_dir()]
    for classPath in instancePathList:
        dataList.append(buildCropDiseaseCountTuple(classPath))

    ## Make a DataFrame
    samplesDataFrame = pd.DataFrame(dataList, columns=['Crop', 'Disease', 'numberImages'])
    ## Save the dataframe
    samplesDataFrame.to_csv(dname.joinpath(csvFilename), index=False)

#%%
#readFolderAndSaveDataFrame("PVD_MenData")

#%%
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("img_folder", help="Folder with Data") 
    parser.add_argument("op_file", help="Output CSV file for data storage") 
    args = parser.parse_args()

    logging.info(f"The folder used is {args.img_folder}")

    readFolderAndSaveDataFrame(args.img_folder, args.op_file)

# %%

# %%
def visualizeCropDiseaseSamples(df, cropName):
    cornDF = df.query(f'Crop == "{cropName}"')
    cornDF.plot(x='Disease', kind='bar', title=f"{cropName}")

#%%
#sampleDF = pd.read_csv('PlantDoc.csv')
#%%
#visualizeCropDiseaseSamples(sampleDF, 'Apple')
# %%
