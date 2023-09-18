import matplotlib.pyplot as plt 
from IPython import display

plt.ion()

def plot(scores, mean_scores, losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    ax1 = plt.subplot(211)
    ax1.set_ylabel('Score')
    ax1.plot(scores,label="Scores")
    ax1.plot(mean_scores, label = "Mean score")
    ax1.set_ylim(ymin=0)
    ax1.legend()
    ax1.text(len(scores)-1,scores[-1],str(scores[-1]))
    ax1.text(len(mean_scores)-1,mean_scores[-1],str(mean_scores[-1]))

    ax2 = plt.subplot(212,sharex = ax1)
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('Huber Loss')
    ax2.plot(losses)
    ax2.set_ylim(ymin=0)
    ax2.text(len(losses)-1,losses[-1],str(losses[-1]))
    plt.show(block=False)
    #plt.pause(.01)