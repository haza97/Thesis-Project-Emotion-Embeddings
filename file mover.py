import shutil
import os

people_list = range(12,21)
#people_list = [11]
for number in people_list:
    emotion_list = ["Sad", "Angry", "Surprise", "Neutral"]
    for emotion in emotion_list:    
        source = [f"D:/ESD/00{number}/{emotion}/evaluation/",
                  f"D:/ESD/00{number}/{emotion}/test/", f"D:/ESD/00{number}/{emotion}/train"]

        dest = f"D:/ESD/00{number}/{emotion}/"

        for folder in source:
            #os.rmdir(folder)
            files = os.listdir(folder)
            for file in files:
                file_path = os.path.join(folder, file)
                shutil.move(file_path, dest)
