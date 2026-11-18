# TriAID - AI Tutor Assistant

## What is TriAID?

Our concept is very simple. Students can get lost or confused while studying a subject, but do not ask for help. We thought of creating a system that would be able to step in if a student is having a hard time and want some help.

We are going to do this in three parts:

1. Computer Vision (James): Looks at a student webcam video and generates some scores on boredom, confusion, etc.
2. Middle Model (Jayden): assembles those scores and attempts to determine if the system should follow up with the student.
3. LLM (Brett): Would give the student some assistance if there was a need to check-in.

It's not that we want to interrupt the moment they get confused, we want it to check-in when it's natural to do so.

## Team Members

| Name | Part of Project |
| --- | --- |
| James Martin | Computer Vision |
| Jayden Allen | Neural Network / GRU Middle Model |
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

- My model uses OpenCV's face detection software as well as MobileNetV2 and a small neural net to output emotional scores.
- To run: open the notebook in Colab w/ the T4 GPU selected and set DEMO_MODE = True -> allows testing without DAiSEE,
  for full use, download DAiSEE (info down below), upload the complete or partial set to your Drive, and run through the code cells.
  IMPORTANT: if run with DEMO_MODE = False, Colab will prompt you to connect your Drive to search for DAiSEE 
- So far, testing has yielded these results: 56.75% accuracy, 0.184 MAE, 8-12 FPS.
- The notebook has extra markdown cells for further clarification.

  While far from perfect (mostly due to DAiSEE confusion & frustration labeling flaws) the code should be in a stable condition. 

### Middle Model - Jayden

So this is my component which is the middle model. Based on these four numbers from the emotion labels, it determines whether the student gets a small check-in. I am using a GRU with 32 hidden units with a simple extra rule.

Here are the main details:

- Inputs: boredom, engagement, confusion and frustration in that order.
- Scores: each is 0-3. I am normalising them by dividing through by 3 for the model.
- Network: 4 inputs, 32 GRU hidden units, Layer Norm, 0.1 dropout, and one yes/no output.
- Training: AdamW, 0.003, 0.03, 70 epochs, Patience 7.
- Testing: five of the student-grouped folds, with groups set aside for tuning and calibration.
- Extra rule: Check in if the GRU score reaches its cutoff, or if confusion or frustration is at least 2.

The other thing to note is that this model only receives one set of four scores per clip (T=1). Though I did incorporate a GRU, this version hasn't learned how a student's emotion fluctuates over time yet. That's something I plan to add in the future.

This output is a small python dictionary with step_in (true, false) and intervention_level. Level 0 is no check in, and level 1 is a gentle check in. We haven't trained it to predict levels 2 or 3.

I got this piece up and running and have run it on the scores of the data set. I also need to run it on the real CV predictions from James and pass the output back to Brett.

### LLM - Brett

Brett is handling the LLM portion, which would reply if the middle model checks in.

- What Model it is and how it works: The LLM is in NLG terms, the surface realisation layer for our overall model. It provides the test taker an interactive tutor through which they may type prompts and receive responses. The GRU passes the LLM a tuple, step_in(true,false) as put earlier. As long as the first value of the tuple is true, the intervention level is considered. The intervention level is from 0-3:
- 0 means stay quiet
- 1 means gentle check in
- 2 means offer a hint
- 3 means offer guided help
The notebook within the instructions to run the model is the baseline model, a standard single-llm approach. This approach uses an  intervention level prompt + system prompt to assist students. My model is called Dynamic Learner Modelling, and aims to change this into: intervention level prompt + intervention-style prompt + system prompt.
The current approach to deriving the intervetion-style prompt consists of four steps. Language is difficult to quantify, so each part requires extensive manual testing. The approach so far is:
1) Difficulty Diagnosis: Classifies the underlying difficulty and provides supporting evidence using a 0-1 scale per subject, stored as a dictionary (Ex conceptual:0.2, syntatic:0.5, logical:0.0).
2) Answer Evaluator: A separate LLM measures the "correctness" of test taker answers to the tutor LLM's responses using a 0-1 scale, as well as justification. 
3) Learner Modelling: Combines answer evaluation and difficulty scores into one data type. Uses an exponential weighed moving average to re-rank results, more recent prompts weigh heavier. Outputs an assistance weight from 0-1.
4) Document Planning: Deterministic, uses the assistance weight to select system-prompts that rate at what level assistance should be given. This intervention-style is added in addition to the one that corresponds with the intervention level. 
   -0-0.25: No additional assistance.
   -0.251-0.5: Small hint, no explaination.
   -0.51-0.75: Mid-level hint, small explaination. 
   -0.751-1: Beginner level explanation. 
6) Surface Realisation: A given LLM generates the final output given the intervention level, assistance, and default system prompt.  

- How to run it: First, you will require an API key and to add that key as a Secret within Google Collab. Assuming this is done, Open the notebook "LLM experiment V1.ipynb" in Google Collab. Run all cells in top-down order. The cell containing "========== INTERACTIVE LLM TESTING ==========" will be where you can conversate with the agent. The very last cell will export the conversation as a CSV for analysis. 
- Results so far: Only the baseline model is integrated enough to produce responses. Not all tokens are compatible with CSV and not all questions receive a response from the model. Nevertheless, the tutor is able to work given a sufficiently large model. 
- Problems or limitations: The Answer evaluator currently uses hardcoded questions and answers, and the scores have been pretty consistent across code and non-code questions and answers. To support free-form conversation however, answer detection, and question detection will need to be implemented. Learner modelling EWMA(exponential weighted moving average) scores are not commensurate with test taker strengths and difficulties. A better algorithm will be required. Finally, true assistance of a test taker requires a corpus of long conversations (conversations exceeding 20 turns) in order for there to be enough data.

## Dataset

We are using DAiSEE, which contains the student videos, and labels of boredom, engagement, confusion, and frustration. All the labels are scored from 0-3.

Dataset page: https://people.iith.ac.in/vineethnb/resources/daisee/index.html

It doesn't tell the model clearly if the tutor should intervene, so we watched 274 of the clips from 69 students in the training set and provided our own check-in labels:

| Our label | Nos. Of clips |
| --- | ---: |
| Check in | 100 |
| Stay quiet | 174 |
| Total | 274 |

We matched the label to the DAiSEE emotion scores using the clip ID. These are our evaluations after watching the clips, not evidence that those students really did need help. 

## What Each Person Worked On

**James Martin:** 
-Built & tested the Computer Vision model
-Uploaded and tested reduced testset of DAiSEE
-Uploaded and implemented full DAiSEE set
-Assisting Jayden with extra labeling

**Jayden Allen:**
- Built and tested the GRU middle model and the extra check-in rule.
- Watched and labeled 274 DAiSEE clips.
- Worked on the labeling setup and the notebook.
- Compared different GRU versions, a simple rule, and other baselines.
- Used student-grouped testing and fixed a class-weight mismatch between training and tuning.
- Worked on moving the notebook to Colab.
- Made the `step_in` / `intervention_level` output for the future LLM connection.

**Brett Raheem:** 
- Build and tested baseline LLM model, providing team-wide API key.
- Assisting with dataset labelling. 
- Developing Dynamic Learner Modelling. 
- Created repo
 

## How to Run
All models were built using Python 3 and can be run in Google Colab. Open the preferred model notebook in Colab (you can find the models on their respected branch), upload the required DAiSEE dataset and `observer_annotations.csv` file to Google Drive, and make sure the file paths match your folders. Once everything is set up, run the notebook cells in order to train and test the model and view the results.
### Dataset Website (Consent Required): https://people.iith.ac.in/vineethnb/resources/daisee/
