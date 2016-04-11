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
		$.each(this.frames, $.proxy(function(index, frame) {
			this.frames[index] = JSON.parse(frame);
		}, this));
	},
	update : function(current){
		if (current === undefined) {
			var data = undefined;
		} else {
			var data = this.frames[current];
		}
		if (data !== undefined) {
			if (data.root !== undefined) {
				var doc = Bokeh.index[data.root].model.document;
			}
			$.each(data.data, function(i, value) {
				if (data.root !== undefined) {
					var ds = doc.get_model_by_id(value.id);
				} else {
					var ds = Bokeh.Collections(value.type).get(value.id);
				}
				if (ds != undefined) {
					ds.set(value.data);
					ds.trigger('change');
				}
			});
		}
	},
	dynamic_update : function(current){
		if (current === undefined) {
			return
		}
		if(this.dynamic) {
			current = JSON.stringify(current);
		}
		function callback(initialized, msg){
			/* This callback receives data from Python as a string
			   in order to parse it correctly quotes are sliced off*/
			if (msg.content.ename != undefined) {
				this.process_error(msg);
			}
			if (msg.msg_type != "execute_result") {
				console.log("Warning: HoloViews callback returned unexpected data for key: (", current, ") with the following content:", msg.content)
				this.time = undefined;
				this.wait = false;
				return
			}
			this.timed = (Date.now() - this.time) * 1.1;
			if (msg.msg_type == "execute_result") {
				if (msg.content.data['text/plain'] === "'Complete'") {
					this.wait = false;
					if (this.queue.length > 0) {
						this.time = Date.now();
						this.dynamic_update(this.queue[this.queue.length-1]);
						this.queue = [];
					}
					return
			    }
				var data = msg.content.data['text/plain'].slice(1, -1);
				this.frames[current] = JSON.parse(data);
				this.update(current);
			}
		}
		var kernel = IPython.notebook.kernel;
		callbacks = {iopub: {output: $.proxy(callback, this, this.initialized)}};
		var cmd = "holoviews.plotting.widgets.NdWidget.widgets['" + this.id + "'].update(" + current + ")";
		kernel.execute("import holoviews;" + cmd, callbacks, {silent : false});
	}
}

// Extend Bokeh widgets with backend specific methods
extend(BokehSelectionWidget.prototype, BokehMethods);
extend(BokehScrubberWidget.prototype, BokehMethods);
