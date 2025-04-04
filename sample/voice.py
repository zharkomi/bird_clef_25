import pickle

# To read (load) a pickle file
with open('/home/mikhail/prj/bird_clef_25/data/train_voice_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Now you can work with the loaded data
print(data)