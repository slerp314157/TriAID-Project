# TriAID - AI Tutor Assistant

## What is TriAID?

Our concept is very simple. Students can get lost or lost in the depth of a subject they study, but do not ask for help. We thought of creating a system that would be able to step in if a student is having a hard time and want some help.

We are going to do this in three parts:

1. Computer Vision (James): Looks at a student webcam video and generates some scores on boredom, confusion, etc.
2. Middle Model (Jayden): assembles those scores and attempts to determine if the system should follow up with the student.
3. LLM (Brett): Would give the student some assistance if there was a need to check-in.

It's not that we want to interrupt the moment they get confused, we want it to check-in when it's natural to do so.

## Team Members

| Name | Partie du projet |
| --- | --- |
| James Martin | Computer Vision |
| Jayden Allen | Neural Network / GRU midd Model |
| Brett Raheem | Large Language Model |

## How It Is Supposed To Work

```text
Student's webcam
|
v
Computer Vision (James)
Gets boredom, engagement, confusion, and frustration scores
|
v
Middle Model (Jayden)
Decides if we should check in
|
v
LLM (Brett)
Is available to support the student
```

This is how everything is going to stitch together. It's not a finished, live pipeline yet.

## Our Models

### Computer Vision - James

James is working on the computer vision portion. This will be what supplies the middle model with emotion scores.

- Model and what it uses: [James: add this]
- How to run it: [James: add this]
- Results so far: [James: add this]
- Problems or limitations: [James: add this]

### Middle Model - Jayden

So this is my component which is the middle model. Based on these four numbers from the emotion labels, it determines whether the student gets a small check-in. I am using a GRU with 32 hidden units with a simple extra rule.

Here are the main details:

- Inputs: boredom, engagement, confusion and frustration in that order.
- Scores: each is 0-3. I am normalising them by dividing through by 3 for the model.
- Network: 4 inputs, 32 GRU hidden units, Layer Norm, 0.1 dropout, and one yes/no output.
- Training: AdamW, 0.003, 0.03, 70 epochs, Patience 7.
- Testing: five of the student-grouped folds, with groups set aside for tuning and calibration.
- Extra rule: check in if the score for either GRU is hit its cutoff, or if the confusion is at least 2, or the frustration is at least 2.

The other thing to note is that this model only receives one set of four scores per clip (T=1). Though I did incorporate a GRU, this version hasn't learned how a student's emotion fluctuates over time yet. That's something I plan to add in the future.

This output is a small python dictionary with stepin (true, false) and interventionlevel. Level 0 is no check in, and level 1 is a gentle check in. We haven't trained it to predict levels 2 or 3.

I got this piece up and running and have run it on the scores of the data set. I also need to run it on the real CV predictions from James and pass the output back to Brett.

### LLM - Brett

Brett is handling the LLM portion, which would reply if the middle model checks in.

- What Model it is and how it works: [Brett: add this]
- How to run it: [Brett: add this]
- Results so far: [Brett: add this]
- Problems or limitations: [Brett: add this]

## Dataset

We are using DAiSEE, which contains the student videos, and labels of boredom, engagement, confusion, and frustration. All the labels are scored from 0-3.

Dataset page: https://people.iith.ac.in/vineethnb/resources/daisee/index.html

It doesn't tell the model the clearly if the tutor should intervene, so we watched 274 of the clips from 69 students in the training set and provided our own check-in labels:

| My label | Nos. Of clips |
| --- | ---: |
| Check in | 100 |
| Stay quiet | 174 |
| Total | 274 |

We matched the label to the DAiSEE emotion scores using the clip ID. These are our evaluations after watching the clips, not evidence that those students really did need help. 

## What Each Person Worked On

**James Martin:** 

**Jayden Allen (me):**
- Built and tested the GRU middle model and the extra check-in rule.
- Watched and labeled 274 DAiSEE clips.
- Worked on the labeling setup and the notebook.
- Compared different GRU versions, a simple rule, and other baselines.
- Used student-grouped testing and fixed a class-weight mismatch between training and tuning.
- Worked on moving the notebook to Colab.
- Made the `step_in` / `intervention_level` output for the future LLM connection.

**Brett Raheem:** 


## How to Run
All models were built using Python 3 and can be run in Google Colab. Open the prefered model notebook in Colab, upload the required DAiSEE dataset and `observer_annotations.csv` file to Google Drive, and make sure the file paths match your folders. Once everything is set up, run the notebook cells in order to train and test the model and view the results.
### Database Website (Consent Required): https://people.iith.ac.in/vineethnb/resources/daisee/
