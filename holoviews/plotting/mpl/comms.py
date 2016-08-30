import uuid
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from matplotlib.backends.backend_nbagg import CommSocket, new_figure_manager_given_figure
except ImportError:
    CommSocket = object
from mpl_toolkits.mplot3d import Axes3D

from ..comms import JupyterComm

mpl_msg_handler = """
target = $('#{comms_target}');
img = $('<div />').html(data);
target.children().each(function () {{ $(this).remove() }})
target.append(img)
"""

mpld3_msg_handler = """
target = $('#fig_el{comms_target}');
target.children().each(function () {{ $(this).remove() }});
mpld3.draw_figure("fig_el{comms_target}", data);
"""

class WidgetCommSocket(CommSocket):
    """
    CustomCommSocket provides communication between the IPython
    kernel and a matplotlib canvas element in the notebook.
    A CustomCommSocket is required to delay communication
    between the kernel and the canvas element until the widget
    has been rendered in the notebook.
    """

    def __init__(self, manager, target=None):
        self.supports_binary = None
        self.manager = manager
        self.target = str(uuid.uuid4()) if target is None else target
        self.html = "<div id=%r></div>" % self.target

    def start(self):
        try:
            # Jupyter/IPython 4.0
            from ipykernel.comm import Comm
        except:
            # IPython <=3.0
            from IPython.kernel.comm import Comm

        try:
            self.comm = Comm('matplotlib', data={'id': self.target})
        except AttributeError:
            raise RuntimeError('Unable to create an IPython notebook Comm '
                               'instance. Are you in the IPython notebook?')
        self.comm.on_msg(self.on_message)
        self.comm.on_close(lambda close_message: self.manager.clearup_closed())



class NbAggJupyterComm(JupyterComm):

    def get_figure_manager(self):
        fig = self._plot.state
        count = self._plot.renderer.counter
        self.manager = new_figure_manager_given_figure(count, fig)

        # Need to call mouse_init on each 3D axis to enable rotation support
        for ax in fig.get_axes():
            if isinstance(ax, Axes3D):
                ax.mouse_init()
        self._comm_socket = WidgetCommSocket(target=self.target,
                                             manager=self.manager)
        return self.manager


    def init(self, on_msg=None):
        if not self._comm:
            self._comm_socket.start()
            self._comm = self._comm_socket.comm
            self.manager.add_web_socket(self._comm_socket)


    def send(self, data):
        if not self._comm:
            self.init()
        self._comm_socket.send_json({'type':'draw'})

