import plotly.graph_objs as go



class ChartFormats():

    def __init__(self) -> None:
        pass

    @staticmethod
    def common_chart_layout():
        return {
            'plot_bgcolor': 'white',  # set plot background color to white
            'paper_bgcolor': 'white',  # set paper background color to white
            'xaxis': {
                'showgrid': False,  # no gridlines
                'linecolor': '#002147',  # navy blue axis line color
                'linewidth': 2,  # thickness of the axis line
            },
            'yaxis': {
                'showgrid': True,
                'linecolor': '#002147',
                'linewidth': 1,
            },
            'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}  # adjust margins to prevent clipping
        }

    @staticmethod
    def get_color_pallete():
        
        # Blue-green gradient palette
        return ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']