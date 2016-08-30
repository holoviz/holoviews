import uuid

import param
from ipykernel.comm import Comm as IPyComm


class Comm(param.Parameterized):
    """
    Comm encompasses any uni- or bi-directional connection between
    a python process and a frontend allowing passing of messages
    between the two. A Comms class must implement methods
    to initialize the connection, send data and handle received
    message events.
    """

    def __init__(self, plot, target=None, on_msg=None):
        """
        Initializes a Comms object
        """
        self.target = target if target else str(uuid.uuid4())
        self._plot = plot
        self._on_msg = on_msg
        self._comm = None


    def init(self, on_msg=None):
        """
        Initializes comms channel.
        """


    def send(self, data):
        """
        Sends data to the frontend
        """


    def decode(self, msg):
        """
        Decode incoming message, e.g. by parsing json.
        """
        return msg


    @property
    def comm(self):
        if not self._comm:
            raise ValueError('Comm has not been initialized')
        return self._comm


    def _handle_msg(self, msg):
        """
        Decode received message before passing it to on_msg callback
        if it has been defined.
        """
        if self._on_msg:
            self._on_msg(self.decode(msg))



class JupyterComm(Comm):
    """
    JupyterComm allows for a bidirectional communication channel
    inside the Jupyter notebook. A JupyterComm requires the comm to be
    registered on the frontend along with a message handler. This is
    handled by the template, which accepts three arguments:

    * comms_target - A unique id to register to register as the comms target.
    * msg_handler -  JS code that processes messages sent to the frontend.
    * init_frame  -  The initial frame to render on the frontend.
    """

    template = """
    <script>
      function msg_handler(msg) {{
        var data = msg.content.data;
        {msg_handler}
      }}

      function register_handler(comm, msg) {{
        comm.on_msg(msg_handler)
      }}

      if ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel !== undefined)) {{
        comm_manager = Jupyter.notebook.kernel.comm_manager
        comm_manager.register_target("{comms_target}", register_handler);
      }}
    </script>

    <div id="{comms_target}">
      {init_frame}
    </div>
    """

    def init(self):
        if self._comm:
            self.warning("Comms already initialized")
            return
        self._comm = IPyComm(target_name=self.target, data={})
        self._comm.on_msg(self._handle_msg)


    def send(self, data):
        """
        Pushes data to comms socket
        """
        if not self._comm:
            raise ValueError("Comm has not been initialized.")
        self._comm.send(data)


    def decode(self, msg):
        return msg['content']['data']
