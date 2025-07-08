#Script used to get the number of classes in train/dev dataset along with the number of unqiue speaker classes

# from sklearn.model_selection import train_test_split

# def get_class_num(folder_dir):
#     speaker_counts = defaultdict(int)
#     speaker_files = defaultdict(list)

#     unique_speakers = set()
#     num_files = 0
    
    
#     for filename in os.listdir(folder_dir):
#         # print(filename)
#         num_files += 1
#         if filename.endswith(".spkid.txt"):
#             filepath = os.path.join(folder_dir, filename)
#             with open(filepath, 'r') as file:
#                 speaker_id = file.read()
#                 # print(int(speaker_id))
#                 int(speaker_id)
                
#                 # if int(speaker_id).isdigit():
#                 # print("hi")
#                 speaker_id = int(speaker_id)
#                 speaker_counts[speaker_id] += 1
#                 unique_speakers.add(speaker_id)
    
#     print(f"Total unique speakers: {len(unique_speakers)}")
#     print("Speaker Counts: ")
#     print("Total Files: ", num_files)
#     for speaker, count in sorted(speaker_counts.items()):
#         print(f"Speaker {speaker}: {count}")
    
#     target_speaker = 627
#     if target_speaker in speaker_files:
#         print(f"\nFiles for Speaker {target_speaker} (up to 10 occurrences):")
#         for file in speaker_files[target_speaker][:10]:
#             print(file)
#     else:
#         print(f"\nSpeaker {target_speaker} not found in dataset.")
    
#     return speaker_counts, len(unique_speakers)

# # directory = "/home/downina3/CSCI481/fin_project/task1/train" #split train dataset first
# # print(directory)
# # get_class_num(directory)

# def split_train_dataset(speaker_dict, )


import os
from collections import defaultdict

def get_speaker_counts(directory):
    speaker_counts = defaultdict(int)    #count occurrences per speaker
    speaker_files = defaultdict(list)    #store filenames per speaker
    
    # dialect_counts = defaultdict(int)
    # dialect_files = defaultdict(list)
    # speaker_to_dialect = {}
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        #look for the speaker id file
        if filename.endswith(".spkid.txt"):
            with open(filepath, 'r') as file:
                speaker_id = file.read().strip()
                
                # print("speaker", speaker_id)
                if speaker_id.isdigit():
                    speaker_id = int(speaker_id)
                    speaker_counts[speaker_id] += 1
                    
                    #store the wav file
                    wav_filename = filename.replace(".spkid.txt", ".wav")
                    speaker_files[speaker_id].append(wav_filename)

        # if filename.endswith(".dialect.txt"):
        #     with open(filepath, 'r') as file:
        #         dialect_id = file.read().strip()
                
        #         if dialect_id.isdigit():
        #             # print("dialect", dialect_id)
        #             dialect_id = int(dialect_id)
        #             dialect_counts[dialect_id] += 1
                    
        #             wav_filename = filename.replace(".dialect.txt", ".wav")
        #             dialect_files[dialect_id].append(wav_filename)

        #             # Extract the file number (e.g., file183 -> 183) to map speaker to dialect
        #             file_number = filename.replace(".dialect.txt", "").replace("file", "")
        #             if file_number.isdigit():
        #                 file_number = int(file_number)
        #                 speaker_to_dialect[file_number] = dialect_id


    for speaker, count in sorted(speaker_counts.items()):
        dialect = speaker_to_dialect.get(speaker, "Unknown")
        print("Speaker", speaker, ":", count, "files") # | Dialect:", dialect

    #print out the 10 audio files for specified target speaker
    
    target_speaker = 628
    print("Files for speaker", target_speaker, ":")
    if target_speaker in speaker_files:
        for file in speaker_files[target_speaker][:10]:
            print(file)
    else:
        print("Not found")
    
    
    print("Total unique speakers:", len(speaker_counts))
    print("Total unique dialects:", len(dialect_counts))


    return speaker_counts, speaker_files, dialect_counts, dialect_files, speaker_to_dialect


directory = "/home/downina3/CSCI481/fin_project/task1/train_fx"
speaker_counts, speaker_files, dialect_counts, dialect_files, speaker_to_dialect = get_speaker_counts(directory)
