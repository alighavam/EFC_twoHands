import argparse
import random
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--subNum', default='101',
                        help='Participant ID (e.g., subj100, subj101, ...)')
    
    args = parser.parse_args()
    subNum = args.subNum

    # choose chords:
    chordIDs = np.array([29212,92122,91211,22911,21291,12129,12291,19111])

    # randomly choose trained and untrained chords:
    trained_chords = np.random.choice(chordIDs, 4, replace=False)
    untrained_chords = np.setdiff1d(chordIDs, trained_chords)

    # choose design parameters:
    nRuns = 8
    nRep = 5 # number of repetition of each chord
    planTime = 0  # time for planning, if 0, go-cue is presented immediately
    execMaxTime = 10000  # maximum allowed time for execution
    feedbackTime = 500  # time to present feedback

    training_days = [2,3,4]
    testing_days = [1,5]

    # target file columns:
    column_names = ['subNum', 'day', 'chordID', 'planTime', 'execMaxTime', 'feedbackTime', 'iti']

    # make the testing sessions target files:
    # quick description of the design -> First 4 runs subject train with the right or left hand, last 4 runs with the other hand.
    #                                    So, we need to have all of the chords in the first and second 4 runs, otherwise one hand might get more of the same chord. 
    #                                    In a day, each chord is visited 5 times for left hand and 5 times for right hand, with 5 reps per visit.     
    for day in testing_days:
        chords = np.concat([trained_chords, untrained_chords])

        # first 4 runs need to have all of the testing chords:
        batch1_trials = np.repeat(chords, 5)            # number of times each chord must be seen
        np.random.shuffle(batch1_trials)                # shuffle the order of chords
        batch1_trials = np.repeat(batch1_trials, nRep)  # this is 8-chords * 5-visits * 5-repetitions = 200 trials

        # last 4 blocks need to have all of the testing chords:
        batch2_trials = np.repeat(chords, 5)
        np.random.shuffle(batch2_trials)
        batch2_trials = np.repeat(batch2_trials, nRep)

        # split the trials into runs:
        runs = np.split(np.concat([batch1_trials,batch2_trials]), nRuns)

        for r, run in enumerate(runs):
            # building the dataframe:
            df = pd.DataFrame(columns=column_names)
            df['chordID'] = run
            df['day'] = np.full_like(run, day)
            df['subNum'] = np.full_like(run, subNum)
            df['planTime'] = np.full_like(run, planTime)
            df['execMaxTime'] = np.full_like(run, execMaxTime)
            df['feedbackTime'] = np.full_like(run, feedbackTime)
            df['iti'] = np.random.randint(500, 1500, len(run))

            # saving the tgt file:
            fname = f'efc2_s{subNum}_day{day:02}_testing_run{r+1}.tgt'
            df.to_csv(os.path.join('target', fname), sep='\t', index=False)
            print(f'{fname} saved!')
        
    # make the training sessions target files:
    # quick description of the design -> For training, only one hand is used. So, trained chords are just shuffled and repeated.   
    #                                    In a day, each chord is visited 20 times, with 5 reps per visit.
    for day in training_days:
        chords = trained_chords

        trials = np.repeat(chords, 20)
        np.random.shuffle(trials)
        trials = np.repeat(trials, nRep)

        # split the trials into runs:
        runs = np.split(trials, nRuns)

        for r, run in enumerate(runs):
            # building the dataframe:
            df = pd.DataFrame(columns=column_names)
            df['chordID'] = run
            df['day'] = np.full_like(run, day)
            df['subNum'] = np.full_like(run, subNum)
            df['planTime'] = np.full_like(run, planTime)
            df['execMaxTime'] = np.full_like(run, execMaxTime)
            df['feedbackTime'] = np.full_like(run, feedbackTime)
            df['iti'] = np.random.randint(500, 1500, len(run))

            # saving the tgt file:
            fname = f'efc2_s{subNum}_day{day:02}_training_run{r+1}.tgt'
            df.to_csv(os.path.join('target', fname), sep='\t', index=False)
            print(f'{fname} saved!')
        

