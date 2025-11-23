import matplotlib.pyplot as plt
from IPython import display

# Create global figure + axis only once
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))

def plot(scores, averages):
    display.clear_output(wait=True)

    ax.clear()  # clear old data from same graph (NOT create new graph)

    ax.set_title("Training Progress")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score")

    ax.plot(scores, label="Scores")
    ax.plot(averages, label="Average Score")

    ax.legend()
    ax.set_ylim(bottom=0)

    if scores:
        ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    if averages:
        ax.text(len(averages)-1, averages[-1], f"{averages[-1]:.2f}")

    fig.canvas.draw()
    fig.canvas.flush_events()
