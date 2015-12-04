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
	init_slider : function(init_val){
		if(this.load_json) {
			var data_url = this.json_path + '/' + this.id + '.json';
			$.getJSON(data_url, $.proxy(function(json_data) {
				this.frames = json_data;
				$.each(this.frames, $.proxy(function(index, frame) {
					this.frames[index] = JSON.parse(frame);
				}, this));
			}, this));
		} else {
			$.each(this.frames, $.proxy(function(index, frame) {
				this.frames[index] = JSON.parse(frame);
			}, this));
		}
	},
	update : function(current){
		if (current === undefined) {
			var data = undefined;
		} else {
			var data = this.frames[current];
		}
		if (data !== undefined) {
			if (data.root !== undefined) {
				doc = Bokeh.index[data.root].model.document;
				$.each(data, function(id, value) {
					ds = doc.get_model_by_id(id);
					if (ds != undefined) {
						ds.set(value.data);
					}
				});
			} else {
				$.each(data, function(id, value) {
					var ds = Bokeh.Collections(value.type).get(id);
					if (ds != undefined) {
						ds.set(value.data);
					}
				});
			}
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
				return
			}
			if (msg.msg_type == "execute_result") {
				var data = msg.content.data['text/plain'].slice(1, -1);
				this.frames[current] = JSON.parse(data);
				this.update(current);
			}
		}
		if(!(current in this.frames)) {
			var kernel = IPython.notebook.kernel;
			callbacks = {iopub: {output: $.proxy(callback, this, this.initialized)}};
			var cmd = "holoviews.plotting.widgets.NdWidget.widgets['" + this.id + "'].update(" + current + ")";
			kernel.execute("import holoviews;" + cmd, callbacks, {silent : false});
		} else {
			this.update(current);
		}
	}
}

// Extend Bokeh widgets with backend specific methods
extend(BokehSelectionWidget.prototype, BokehMethods);
extend(BokehScrubberWidget.prototype, BokehMethods);
