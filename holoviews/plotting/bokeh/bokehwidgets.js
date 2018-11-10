// Define Bokeh specific subclasses
function BokehSelectionWidget() {
  SelectionWidget.apply(this, arguments);
}

function BokehScrubberWidget() {
  ScrubberWidget.apply(this, arguments);
}

// Let them inherit from the baseclasses
BokehSelectionWidget.prototype = Object.create(SelectionWidget.prototype);
BokehScrubberWidget.prototype = Object.create(ScrubberWidget.prototype);

// Define methods to override on widgets
var BokehMethods = {
  update_cache : function(){
    for (var index in this.frames) {
      this.frames[index] = JSON.parse(this.frames[index]);
    }
  },
  update : function(current){
    if (current === undefined) {
      return;
    }
    var data = this.frames[current];
    if (data !== undefined) {
      if (data.root in HoloViews.plot_index) {
        var doc = HoloViews.plot_index[data.root].model.document;
      } else {
        var doc = Bokeh.index[data.root].model.document;
      }
      doc.apply_json_patch(data.content);
    }
  },
  init_comms: function() {
    if (Bokeh.protocol !== undefined) {
      this.receiver = new Bokeh.protocol.Receiver()
    } else {
      this.receiver = null;
    }
    return HoloViewsWidget.prototype.init_comms.call(this);
  },
  process_msg : function(msg) {
    if (this.plot_id in HoloViews.plot_index) {
      var doc = HoloViews.plot_index[this.plot_id].model.document;
    } else {
      var doc = Bokeh.index[this.plot_id].model.document;
    }
    if (this.receiver === null) { return }
    var receiver = this.receiver;
    if (msg.buffers.length > 0) {
      receiver.consume(msg.buffers[0].buffer)
    } else {
      receiver.consume(msg.content.data)
    }
    const comm_msg = receiver.message;
    if ((comm_msg != null) && (doc != null)) {
      doc.apply_json_patch(comm_msg.content, comm_msg.buffers)
    }
  }
}

// Extend Bokeh widgets with backend specific methods
extend(BokehSelectionWidget.prototype, BokehMethods);
extend(BokehScrubberWidget.prototype, BokehMethods);

window.HoloViews.BokehSelectionWidget = BokehSelectionWidget
window.HoloViews.BokehScrubberWidget = BokehScrubberWidget
