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

    template = ''

    def __init__(self, plot, id=None, on_msg=None):
        """
        Initializes a Comms object
        """
        self.id = id if id else uuid.uuid4().hex
        self._plot = plot
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

    template = """
    <script>
      function msg_handler(msg) {{
        var msg = msg.content.data;
        {msg_handler}
      }}

      if ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel != null)) {{
        comm_manager = Jupyter.notebook.kernel.comm_manager;
        comm_manager.register_target("{comm_id}", function(comm) {{ comm.on_msg(msg_handler);}});
      }}
    </script>

    <div id="fig_{comm_id}">
      {init_frame}
    </div>
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

    template = """
    <script>
      function msg_handler(msg) {{
        var msg = msg.content.data;
        {msg_handler}
      }}

      if ((window.Jupyter !== undefined) && (Jupyter.notebook.kernel != null)) {{
        var comm_manager = Jupyter.notebook.kernel.comm_manager;
        comm = comm_manager.new_comm("{comm_id}", {{}}, {{}}, {{}}, "{comm_id}");
        comm.on_msg(msg_handler);
      }}
    </script>

    <div id="fig_{comm_id}">
      {init_frame}
    </div>
    """

    def __init__(self, plot, id=None, on_msg=None):
        """
        Initializes a Comms object
        """
        from IPython import get_ipython
        super(JupyterCommJS, self).__init__(plot, id, on_msg)
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

