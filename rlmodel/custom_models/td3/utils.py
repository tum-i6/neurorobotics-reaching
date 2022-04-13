import matplotlib.pyplot as plt   # for plotting of results

def rescale(input, out_range, in_range):
    """ Rescale the input to a given output range
        
        Args:
            input (array): current state of environment
            
        Returns:
            action (float): action clipped within action range
        
        """
    input_range = in_range[1] - in_range[0]
    output_range = out_range[1] - out_range[0]
    return output_range * (input - in_range[0]) / input_range + out_range[0]

def scatter_plotter(data, show_plot=False):
    """ Create a scatter plot for the robotic reaching task showing which postions did the robot reach and which not

        Args:
            data (dict): Dictionary containing the x and y positions of the points, the color c of the points and the title of the plot
            show_plot (boolean): Whether to show the created scatter plot
        
        Returns:
            fig: Scatter plot figure created from the input data
    """

    fig, ax = plt.subplots()
    ax.scatter(data["x"], data["y"], c=data["c"])
    ax.set_title(data["title"])
    ax.set(xlabel='x-coord [m]', ylabel='y-coord [m]')
    ax.set_xlim((-0.44,0.48))
    ax.set_ylim((-0.48,0.44))

    if show_plot==True:
        plt.show()
    else:
        return fig