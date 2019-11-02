import torch
import numpy as np
import os
import sys
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel
import time


roberta = RobertaModel.from_pretrained('checkpoints/', checkpoint_file='ck.pt', data_name_or_path='data/processed_RACE/')
roberta.eval()



def eval_one_example():
    # context = 'I was not very happy. Because he did some bad things to me. But I am fine after he apologized to me.'
    # qa1 = 'What\'s my mood right now? Pleased'
    # qa2 = 'What\'s my mood right now? Sad'
    # qa3 = 'What\'s my mood right now? Angry'
    # qa4 = 'What\'s my mood right now? Cool'

    context = 'The Sunset Pasta Cruise to Emerald Bay Saturday evening, September 25, 2010 You will cruise to Emerald Bay at Sunset, one of the most beautiful places in the world while dining on a Pasta Buffet and listening to live light dance music. Buses will pick up Sunset Pasta Cruise diners from the main entrance to the Horizon Casino Resort at: 4:40pm and 5:05pm on Saturday and take you the 1.5 miles to Ski Run Marina for boarding. Boarding is at Ski Run Marina at 5:15 p.m. (with departure at 5:45 p.m.), located in South Lake Tahoe. The cost for the cruise, pasta buffet, live music, and the 2.5-hour cruise to Emerald Bay is $55 (normally $75). The cost for children between 3-11 is $41 and under 3 is free. Must register the under 3 as well for the coast guard count. The Sunset Pasta Cruise will be limited to 200 guests. Large parties will be seated first to insure seating together. Pick up your Sunset Pasta Cruise tickets at the Expo at the Horizon Casino Resort before 3 p.m. on Saturday. Those unclaimed will be sold to those on the waiting list at that time. At approximately 5:45 pm any extra spaces will be sold to passengers on the dock. Children who require a seat must have a ticket as well. Closest lodging to the Pasta Cruise is: Super 8, Lakeland Village. Please note that our sponsor , the Riva Grill, is on the Lake close to the boarding area for the Tahoe Queen. A great gathering place to meet or to have dinner. Call Riva Grill (530) 542-2600 for lunch or dinner reservations while you are visiting Lake Tahoe.'

    qas = ['When will the cruise to Emerald Bay end? At about 7:00 pm.', 'When will the cruise to Emerald Bay end? At about 8:20 pm.', 'When will the cruise to Emerald Bay end? At about 9:20 pm.', 'When will the cruise to Emerald Bay end? On Sunday morning.']
    t1 = time.time()
    ans = 1
    ts = []
    for qa in qas:
        inp = roberta.encode(qa, context)
        ts.append(inp)
    batch = collate_tokens(ts, pad_idx=1)

    logits = roberta.predict('sentence_classification_head', batch, return_logits=True).tolist()

    logits = np.asarray(logits).flatten()

    print(logits)
    # assert np.argmax(logits) == ans
    t2 = time.time()
    print("Time cost: {}s".format(t2 - t1))

def eval_on_test_set(testset='high'):
    dirpath = 'data/extracted_RACE'
    with open(os.path.join(dirpath, 'test-{}.input0').format(testset)) as fr0, open(os.path.join(dirpath, 'test-{}.input1').format(testset)) as fr1, open(os.path.join(dirpath, 'test-{}.input2').format(testset)) as fr2, open(os.path.join(dirpath, 'test-{}.input3').format(testset)) as fr3, open(os.path.join(dirpath, 'test-{}.input4').format(testset)) as fr4, open(os.path.join(dirpath, 'test-{}.label').format(testset)) as fr5:
        preds = []
        labels = []
        i = 0
        for context, qa1, qa2, qa3, qa4, label in zip(fr0, fr1, fr2, fr3, fr4, fr5):
            ts = []
            for qa in [qa1, qa2, qa3, qa4]:
                inp = roberta.encode(qa.strip(), context.strip().replace('\n', ' '))
                if len(inp) > 512:
                    break
                ts.append(inp)
            if len(ts) != 4:
                continue
            batch = collate_tokens(ts, pad_idx=1)
            logits = roberta.predict('sentence_classification_head', batch, return_logits=True).tolist()
            logits = np.asarray(logits).flatten()
            pred = np.argmax(logits)
            labels.append(int(label.strip()))
            preds.append(pred)
            i += 1
            if i % 1000 == 0:
                print("Finished {} samples.".format(i))
        print(preds)
        print(labels)
        print('Accuracy:', np.mean(np.asarray(preds) == np.asarray(labels)))



eval_one_example()
# eval_on_test_set()
# eval_on_test_set('middle')
