import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from pick import pick


class Visualizer:
    """
    Visualize pd.DataFrame using 2D and 3D plots

    ...
    Methods
    -------
    show_cols
        show 2d plot of multiple columns with shared x axis
    show_cols_3d
        show 3d plot
    show_matrix
        for list of columns, show 2d plot for every pair 
    custom_show
        create custom plot based on user input
    """

    def __init__(self,
                 data: pd.DataFrame,
                 desc: dict,
                 details: dict):
        """Instantiate Visualizer

        Args:
            data (pd.DataFrame): Data to be displayed
            desc (dict): Simple description of columns
            details (dict): Detailed description of df columns
        """
        self._data = data
        self._desc = desc
        self._details = details

    def show_cols(self,
                  title: str,
                  cols: list,
                  x: str,
                  size: str,
                  size_scale: float,
                  color: str,
                  show=True) -> None:
        """Display n vertical columns with shared x axis and color scale

        Args:
            title (str): Plot title
            cols (list): List of columns to be plotted
            x (str): Shared x axis
            size (str): Size, must be integer
            size_scale (float): Scale value for size
            color (str): Shared color
            show (bool): Show plot (default is true)
        """
        # check values
        if size:
            if not self.__valid(*cols, x, size, color):
                return
        else:
            if not self.__valid(*cols, x, color):
                return

        if len(cols) < 1 or not isinstance(cols, (list, tuple, np.ndarray)):
            print("Must have at least 1 column")
            return False
        # create subplot dimensions
        # -1 for x and -1 for color
        fig = make_subplots(
            rows=len(cols), cols=1,
            vertical_spacing=0.05,
            subplot_titles=[self._details[val] for val in cols],
            shared_xaxes=True
        )

        x_data = self._data.loc[:, x]

        for row, col in enumerate(cols, start=1):
            col_data = self._data.loc[:, col]

            # add size if it is not None
            marker_dict = dict(
                color=self._data.loc[:, color],
                coloraxis='coloraxis'
            )
            if size:
                marker_dict.update(dict(
                    size=self._data.loc[:, size],
                    sizemode="area",
                    sizeref=size_scale
                ))

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=col_data,
                    mode='markers',
                    marker=marker_dict
                ),
                row=row, col=1
            )

        # Update axis properties
        fig.update_xaxes(
            title_text=self._details[x], row=(len(cols)), col=1)
        row = 1
        for row, col in enumerate(cols, start=1):
            fig.update_yaxes(title_text=self._desc[col], row=row, col=1)

        # update layout
        fig.update_layout(
            title_text=title,
            coloraxis=dict(colorscale='Turbo'),
            showlegend=False)
        # set color for coloraxis
        fig.update_coloraxes(colorbar_title=dict(text=self._desc[color]))

        if show:
            fig.show()

    def __valid(self, *vals: str) -> bool:
        """Determine if values will produce valid plot
        Args:
            *vals (str): values to be checked
        Returns:
            bool: True if all values are a column in _data, false otherwhise
        """
        try:
            for val in vals:
                if val not in self._data.columns:
                    print(val, "does not exist in data")
                    return False
        except:
            return False
        return True

    def show_cols_3d(self,
                     x: str,
                     y: str,
                     z: str,
                     color: str,
                     hover_title: str,
                     title: str,
                     show=True) -> None:
        """Show 3D plot

        Args:
            x (str): x dimension
            y (str): y dimension
            z (str): z dimension
            color (str): color dimension
            hover_title (str): name of labels for data points
            title (str): graph title
            show (bool): Show plot (default is true)
        """
        # check values
        if not self.__valid(x, y, z, color, hover_title):
            return

        fig = px.scatter_3d(
            self._data,
            x=x,
            y=y,
            z=z,
            color=color,
            color_continuous_scale='turbo',
            labels=self._desc,
            hover_name=hover_title,
            title=title)

        fig.update_layout(
            scene_aspectratio=dict(x=6, y=2, z=2),
            scene_camera=dict(eye=dict(x=0, y=-0.5, z=2)))
        fig.update_xaxes(autorange='reversed')
        if show:
            fig.show()

    def show_matrix(
            self,
            cols: list,
            title: str,
            hover_title: str,
            color: str,
            show=True) -> None:
        """Display data matrix (cartesian comparison)

        Args:
            cols (list): columns to be compared
            title (str): title of graph
            hover_title (str): name of labels for data points
            color (str): color dimension
            show (bool): Show plot (default is true)
        """
        # check values
        if not self.__valid(*cols, hover_title, color):
            return

        # in rare case of user wanting
        # plot of 'stat' x 'stat' with 'stat' coloring,
        # resulting in empty plot
        if isinstance(cols, str):
            cols = [cols]
        if len(cols) == 1 and color in cols:
            dimensions = [color]
        else:
            dimensions = [col for col in cols if col != color]

        fig = px.scatter_matrix(
            self._data,
            dimensions,
            labels=self._desc,
            title=title,
            color=color,
            hover_name=hover_title,
            color_continuous_scale='turbo')
        if show:
            fig.show()

    def __custom_matrix(
            self,
            options: list,
            hover_name,
            name: str,
            show=True) -> None:
        """Plot custom matrix based on user input

        Args:
            options (list): List of space-padded options (col_name[space]+long description...)
            hover_name (str or index): Name for labels on hover
            name (str): Name of data being plotted
            show (bool): Show plot (default is true)
        """

        params = pick(options, 'Pick matrix parameters:',
                      multiselect=True, min_selection_count=1)

        # .partition removes added description
        params = [x[0].partition(' ')[0] for x in params]

        color = pick(options, 'Pick color parameter:')
        color = color[0].partition(' ')[0]
        self.show_matrix(
            params,
            'title',
            hover_name,
            color,
            show)
        return

    def __custom_scatter(
            self,
            options: list,
            is_space: bool,
            hover_name,
            name: str,
            show=True) -> None:
        """Plot custom scatter (2 or 3 dimensional) based on user input

        Args:
            options (list): List of space-padded options (col_name[space]+long description...)
            is_space (bool): true for 3d, false otherwhise
            hover_name (str or index): Name for labels on hover
            name (str): Name of data being plotted
            show (bool): Show plot (default is true)
        """
        x = pick(options, 'Pick x axis:')

        if not is_space:
            y_axes = pick(options, 'Pick y axes (all sharing x axis):',
                          multiselect=True,
                          min_selection_count=1)
            color = pick(options, 'Pick color parameter:')

            y_axes = [val[0].partition(' ')[0] for val in y_axes]
            color = color[0].partition(' ')[0]
            x = x[0].partition(' ')[0]

            self.show_cols(
                'Custom 2D scatter of {}'.format(name),
                y_axes,
                x,
                None,
                1,
                color,
                show
            )
            return

        y = pick(options, 'Pick y axis:')
        z = pick(options, 'Pick z axis')
        color = pick(options, 'Pick color parameter:')

        params = (x, y, z, color)
        params = [val[0].partition(' ')[0] for val in params]

        self.show_cols_3d(
            params[0],
            params[1],
            params[2],
            params[3],
            hover_name,
            'Custom scatter of {} for {} x {} x {}'.format(
                name, self._desc[params[0]], self._desc[params[1]], self._desc[params[2]]),
            show
        )

    def custom_show(self, name: str, show=True) -> None:
        """Let user pick dimensions for either 2D, 3D or matrix plot

        Args:
            name (str): Name of data being plotted
            show (bool, optional): If true, print plot (default True)
        """

        title = 'Pick graph type:'
        graph_type = pick(['2D scatter', '3D scatter',
                           'n-Matrix', 'cancel'], title)

        if graph_type[0] == 'cancel':
            return

        if 'nice_codes' in self._data.columns:
            hover_name = 'nice_codes'
        else:
            hover_name = 'name'

        options = list(self._data.columns)

        len_array = [len(x) for x in options]
        space_count = max(len_array) + 3

        options = [
            val + ' ' * (space_count - len(val)
                         ) + self._details[val]
            for val in options]

        if graph_type[0] == 'n-Matrix':
            self.__custom_matrix(
                options,
                hover_name,
                name,
                show)
            return

        self.__custom_scatter(
            options,
            graph_type[0] == '3D scatter',
            hover_name,
            name,
            show)

    def show_comparison(
            self,
            x: str,
            y: str,
            color: str,
            size: str,
            size_scale: float,
            comp_data: pd.DataFrame,
            title: str,
            show: bool):
        # to do: check values

        fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.1,
            subplot_titles=(self._details[y], self._details[y]),
            shared_xaxes=False
        )

        for row, data in enumerate((self._data, comp_data), start=1):

            x_data = data.loc[:, x]
            y_data = data.loc[:, y]

            # add size if it is not None
            marker_dict = dict(
                color=data.loc[:, color],
                coloraxis='coloraxis'
            )
            if size:
                marker_dict.update(dict(
                    size=data.loc[:, size],
                    sizemode="area",
                    sizeref=size_scale
                ))

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=marker_dict
                ),
                row=row, col=1
            )

        # Update axis properties
        fig.update_xaxes(
            title_text=self._details[x], row=2, col=1)

        for row in range(1, 3):
            fig.update_yaxes(title_text=self._desc[y], row=row, col=1)

        # update layout
        fig.update_layout(
            title_text=title,
            coloraxis=dict(colorscale='Turbo'),
            showlegend=False)
        # set color for coloraxis
        fig.update_coloraxes(colorbar_title=dict(text=self._desc[color]))

        if show:
            fig.show()
