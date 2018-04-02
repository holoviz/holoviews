import json
import uuid
import sys
import traceback

try:
    from StringIO import StringIO
except:
    from io import StringIO


class StandardOutput(list):
    """
    Context manager to capture standard output for any code it
    is wrapping and make it available as a list, e.g.:

    >>> with StandardOutput() as stdout:
    ...   print('This gets captured')
    >>> print(stdout[0])
    This gets captured
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


class Comm(object):
    """
    Comm encompasses any uni- or bi-directional connection between
    a python process and a frontend allowing passing of messages
    between the two. A Comms class must implement methods
    send data and handle received message events.

    If the Comm has to be set up on the frontend a template to
    handle the creation of the comms channel along with a message
    handler to process incoming messages must be supplied.

    The template must accept three arguments:

    * id          -  A unique id to register to register the comm under.
    * msg_handler -  JS code which has the msg variable in scope and
                     performs appropriate action for the supplied message.
    * init_frame  -  The initial frame to render on the frontend.
    """

    html_template = """
    <div id="fig_{plot_id}">
      {init_frame}
    </div>
    """

    js_template = ''

    def __init__(self, id=None, on_msg=None):
        """
        Initializes a Comms object
        """
        self.id = id if id else uuid.uuid4().hex
        self._on_msg = on_msg
        self._comm = None


    def init(self, on_msg=None):
        """
        Initializes comms channel.
        """


    def send(self, data=None, buffers=[]):
        """
        Sends data to the frontend
        """


    @classmethod
    def decode(cls, msg):
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
        comm_id = None
        try:
            stdout = []
            msg = self.decode(msg)
            comm_id = msg.pop('comm_id', None)
            if self._on_msg:
                # Comm swallows standard output so we need to capture
                # it and then send it to the frontend
                with StandardOutput() as stdout:
                    self._on_msg(msg)
        except Exception as e:
            frame =traceback.extract_tb(sys.exc_info()[2])[-2]
            fname,lineno,fn,text = frame
            error_kwargs = dict(type=type(e).__name__, fn=fn, fname=fname,
                                line=lineno, error=str(e))
            error = '{fname} {fn} L{line}\n\t{type}: {error}'.format(**error_kwargs)
            if stdout:
                stdout = '\n\t'+'\n\t'.join(stdout)
                error = '\n'.join([stdout, error])
            reply = {'msg_type': "Error", 'traceback': error}
        else:
            stdout = '\n\t'+'\n\t'.join(stdout) if stdout else ''
            reply = {'msg_type': "Ready", 'content': stdout}

        # Returning the comm_id in an ACK message ensures that
        # the correct comms handle is unblocked
        if comm_id:
            reply['comm_id'] = comm_id
        self.send(json.dumps(reply))


class JupyterComm(Comm):
    """
    JupyterComm provides a Comm for the notebook which is initialized
    the first time data is pushed to the frontend.
    """

    js_template = """
    function msg_handler(msg) {{
      var buffers = msg.buffers;
      var msg = msg.content.data;
      {msg_handler}
    }}
    HoloViews.comm_manager.register_target('{plot_id}', '{comm_id}', msg_handler);
    """

    def init(self):
        from ipykernel.comm import Comm as IPyComm
        if self._comm:
            return
        self._comm = IPyComm(target_name=self.id, data={})
        self._comm.on_msg(self._handle_msg)


    @classmethod
    def decode(cls, msg):
        """
        Decodes messages following Jupyter messaging protocol.
        If JSON decoding fails data is assumed to be a regular string.
        """
        return msg['content']['data']


    def send(self, data=None, buffers=[]):
        """
        Pushes data across comm socket.
        """
        if not self._comm:
            self.init()
        self.comm.send(data, buffers=buffers)



class JupyterCommJS(JupyterComm):
    """
    JupyterCommJS provides a comms channel for the Jupyter notebook,
    which is initialized on the frontend. This allows sending events
    initiated on the frontend to python.
    """

    js_template = """
    <script>
      function msg_handler(msg) {{
        var msg = msg.content.data;
        var buffers = msg.buffers
        {msg_handler}
      }}
      comm = HoloViews.comm_manager.get_client_comm("{comm_id}");
      comm.on_msg(msg_handler);
    </script>
    """

    def __init__(self, id=None, on_msg=None):
        """
        Initializes a Comms object
        """
        from IPython import get_ipython
        super(JupyterCommJS, self).__init__(id, on_msg)
        self.manager = get_ipython().kernel.comm_manager
        self.manager.register_target(self.id, self._handle_open)


    def _handle_open(self, comm, msg):
        self._comm = comm
        self._comm.on_msg(self._handle_msg)


    def send(self, data=None, buffers=[]):
        """
        Pushes data across comm socket.
        """
        self.comm.send(data, buffers=buffers)



class CommManager(object):
    """
    The CommManager is an abstract baseclass for establishing
    websocket comms on the client and the server.
    """

    js_manager = """
    function CommManager() {
    }

    CommManager.prototype.register_target = function() {
    }

    CommManager.prototype.get_client_comm = function() {
    }

    window.HoloViews.comm_manager = CommManager()
    """

    _comms = {}

    server_comm = Comm

    client_comm = Comm

    @classmethod
    def get_server_comm(cls, on_msg=None, id=None):
        comm = cls.server_comm(id, on_msg)
        cls._comms[comm.id] = comm
        return comm

    @classmethod
    def get_client_comm(cls, on_msg=None, id=None):
        comm = cls.client_comm(id, on_msg)
        cls._comms[comm.id] = comm
        return comm



class JupyterCommManager(CommManager):
    """
    The JupyterCommManager is used to establishing websocket comms on
    the client and the server via the Jupyter comms interface.

    There are two cases for both the register_target and get_client_comm
    methods: one to handle the classic notebook frontend and one to
    handle JupyterLab. The latter case uses the globally available
    HoloViews object which is made available when the HoloViews notebook
    extension is loaded. This object is handled in turn by the
    JupyterLab extension which keeps track of the kernels associated
    with each plot, ensuring the corresponding comms can be accessed.
    """

    js_manager = """
    function JupyterCommManager() {
    }

    JupyterCommManager.prototype.register_target = function(plot_id, comm_id, msg_handler) {
      if (window.comm_manager || ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel != null))) {
        var comm_manager = window.comm_manager || Jupyter.notebook.kernel.comm_manager;
        comm_manager.register_target(comm_id, function(comm) {
          comm.on_msg(msg_handler);
        });
      } else if ((plot_id in HoloViews.kernels) && (HoloViews.kernels[plot_id])) {
        HoloViews.kernels[plot_id].registerCommTarget(comm_id, function(comm) {
          comm.onMsg = msg_handler;
        });
      }
    }

    JupyterCommManager.prototype.get_client_comm = function(plot_id, comm_id, msg_handler) {
      if (comm_id in window.HoloViews.comms) {
        return HoloViews.comms[comm_id];
      } else if (window.comm_manager || ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel != null))) {
        var comm_manager = window.comm_manager || Jupyter.notebook.kernel.comm_manager;
        var comm = comm_manager.new_comm(comm_id, {}, {}, {}, comm_id);
        if (msg_handler) {
          comm.on_msg(msg_handler);
        }
      } else if ((plot_id in HoloViews.kernels) && (HoloViews.kernels[plot_id])) {
        var comm = HoloViews.kernels[plot_id].connectToComm(comm_id);
        comm.open();
        if (msg_handler) {
          comm.onMsg = msg_handler;
        }
      }
      HoloViews.comms[comm_id] = comm;
      return comm;
    }

    window.HoloViews.comm_manager = new JupyterCommManager();
    """

    server_comm = JupyterComm

    client_comm = JupyterCommJS
